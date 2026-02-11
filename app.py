"""
Slack Draft Agent - Main Application.

Listens for incoming Slack messages and automatically generates
draft replies using Claude, powered by your full Slack history context.

Features:
- Real-time indexing of ALL messages the bot can see
- 60-day rolling window with automatic daily cleanup
- Recency-weighted semantic search for context
- Cross-channel intelligence for smarter drafts
- Dual delivery: ephemeral in-thread + DM from bot

Usage:
    python app.py
"""
import json
import logging
import re
import time
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from apscheduler.schedulers.background import BackgroundScheduler

import config
import rag
import drafter
from flask import Flask
from threading import Thread
import os

# ── Setup ────────────────────────────────────────────────────────
config.validate()
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = App(token=config.SLACK_BOT_TOKEN, signing_secret=config.SLACK_SIGNING_SECRET)
slack = WebClient(token=config.SLACK_BOT_TOKEN)

# ── User Cache ───────────────────────────────────────────────────
_user_cache = {}


def get_user_name(user_id):
    """Look up a Slack user's display name (cached)."""
    if user_id in _user_cache:
        return _user_cache[user_id]
    try:
        result = slack.users_info(user=user_id)
        profile = result["user"]["profile"]
        name = profile.get("display_name") or profile.get("real_name") or user_id
        _user_cache[user_id] = name
        return name
    except SlackApiError:
        _user_cache[user_id] = user_id
        return user_id


def get_channel_name(channel_id):
    """Look up a channel's name."""
    try:
        result = slack.conversations_info(channel=channel_id)
        ch = result["channel"]
        if ch.get("is_im"):
            return "DM-" + get_user_name(ch.get("user", ""))
        return ch.get("name", channel_id)
    except SlackApiError:
        return channel_id


# ── Thread Context ───────────────────────────────────────────────
def get_thread_messages(channel_id, thread_ts, limit=10):
    """Fetch recent messages from a thread for context."""
    try:
        result = slack.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            limit=limit,
        )
        messages = []
        for msg in result.get("messages", []):
            messages.append({
                "user_name": get_user_name(msg.get("user", "unknown")),
                "text": msg.get("text", ""),
            })
        return messages
    except SlackApiError as e:
        logger.warning("Could not fetch thread: %s", e)
        return []


# ── Helpers ──────────────────────────────────────────────────────
_dm_channel_id = None


def get_dm_channel():
    """Get or open the DM channel between the bot and the user."""
    global _dm_channel_id
    if _dm_channel_id:
        return _dm_channel_id
    try:
        result = slack.conversations_open(users=[config.MY_SLACK_USER_ID])
        _dm_channel_id = result["channel"]["id"]
        return _dm_channel_id
    except SlackApiError as e:
        logger.error("Failed to open DM channel: %s", e)
        return None


def get_permalink(channel_id, message_ts):
    """Get a permalink to a specific Slack message."""
    try:
        result = slack.chat_getPermalink(channel=channel_id, message_ts=message_ts)
        return result.get("permalink", "")
    except SlackApiError:
        return ""


def delete_dm_message(dm_channel, dm_ts):
    """Delete a DM message by its channel and timestamp."""
    if not dm_channel or not dm_ts:
        return
    try:
        slack.chat_delete(channel=dm_channel, ts=dm_ts)
        logger.info("Deleted DM draft message.")
    except SlackApiError as e:
        logger.warning("Could not delete DM message: %s", e)


# ── Draft Delivery (Dual: Ephemeral + DM) ───────────────────────
def send_draft_both(channel_id, thread_ts, draft, original_text, sender_name, channel_name):
    """
    Send the draft two ways:
    1. Ephemeral in-thread (quick access while you're in the channel)
    2. DM from bot (so you never miss it, visible in Threads tab too)

    Both include the same buttons. When either is acted on, the DM is deleted.
    """
    dm_channel = get_dm_channel()
    dm_ts = None

    # ── 1. Send DM first (so we get its ts to link in the ephemeral) ──
    if dm_channel:
        permalink = get_permalink(channel_id, thread_ts)
        if permalink:
            header_text = "💬 *" + sender_name + "* in *#" + channel_name + "* (<" + permalink + "|view message>):\n>" + original_text
        else:
            header_text = "💬 *" + sender_name + "* in *#" + channel_name + "*:\n>" + original_text

        dm_blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": header_text},
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "✏️ *Draft reply:*\n\n" + draft},
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Send"},
                        "style": "primary",
                        "action_id": "send_draft",
                        "value": json.dumps({
                            "channel": channel_id,
                            "thread_ts": thread_ts,
                            "draft": draft,
                            "dm_channel": dm_channel,
                            "dm_ts": "",  # will be updated after post
                        }),
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🔄 Regenerate"},
                        "action_id": "regenerate_draft",
                        "value": json.dumps({
                            "channel": channel_id,
                            "thread_ts": thread_ts,
                            "original_text": original_text,
                            "sender_name": sender_name,
                            "channel_name": channel_name,
                            "dm_channel": dm_channel,
                            "dm_ts": "",
                        }),
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Dismiss"},
                        "action_id": "dismiss_draft",
                        "value": json.dumps({
                            "dm_channel": dm_channel,
                            "dm_ts": "",
                        }),
                    },
                ],
            },
        ]

        try:
            result = slack.chat_postMessage(
                channel=dm_channel,
                text="Draft reply for " + sender_name + " in #" + channel_name + ": " + draft,
                blocks=dm_blocks,
            )
            dm_ts = result["ts"]
            logger.info("Draft sent via DM (ts=%s).", dm_ts)

            # Now update the DM message buttons with the actual dm_ts
            # so clicking buttons on the DM itself can self-delete
            for block in dm_blocks:
                if block.get("type") == "actions":
                    for elem in block["elements"]:
                        val = json.loads(elem["value"])
                        val["dm_ts"] = dm_ts
                        elem["value"] = json.dumps(val)

            slack.chat_update(
                channel=dm_channel,
                ts=dm_ts,
                text="Draft reply for " + sender_name + " in #" + channel_name + ": " + draft,
                blocks=dm_blocks,
            )
        except SlackApiError as e:
            logger.error("Failed to send DM draft: %s", e)

    # ── 2. Send ephemeral in-thread ──
    ephemeral_blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "✏️ *Draft reply:*\n\n" + draft},
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Send"},
                    "style": "primary",
                    "action_id": "send_draft",
                    "value": json.dumps({
                        "channel": channel_id,
                        "thread_ts": thread_ts,
                        "draft": draft,
                        "dm_channel": dm_channel or "",
                        "dm_ts": dm_ts or "",
                    }),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "🔄 Regenerate"},
                    "action_id": "regenerate_draft",
                    "value": json.dumps({
                        "channel": channel_id,
                        "thread_ts": thread_ts,
                        "original_text": original_text,
                        "sender_name": sender_name,
                        "channel_name": channel_name,
                        "dm_channel": dm_channel or "",
                        "dm_ts": dm_ts or "",
                    }),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ Dismiss"},
                    "action_id": "dismiss_draft",
                    "value": json.dumps({
                        "dm_channel": dm_channel or "",
                        "dm_ts": dm_ts or "",
                    }),
                },
            ],
        },
    ]

    try:
        slack.chat_postEphemeral(
            channel=channel_id,
            user=config.MY_SLACK_USER_ID,
            thread_ts=thread_ts,
            text="Draft reply: " + draft,
            blocks=ephemeral_blocks,
        )
        logger.info("Draft also sent as ephemeral in-thread.")
    except SlackApiError as e:
        logger.error("Failed to send ephemeral draft: %s", e)


# ── Index New Messages ───────────────────────────────────────────
def index_message(text, user_id, channel_id, ts):
    """Add a new incoming message to the vector store for future context."""
    try:
        channel_name = get_channel_name(channel_id)
        user_name = get_user_name(user_id)
        rag.add_messages([{
            "id": channel_id + "_" + ts,
            "text": text,
            "user_id": user_id,
            "user_name": user_name,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "timestamp": ts,
        }])
    except Exception as e:
        logger.warning("Failed to index message: %s", e)


# ── Background Scheduler ────────────────────────────────────────
def daily_cleanup():
    """Run daily cleanup of messages older than retention period."""
    logger.info("Running scheduled cleanup (retention: %d days)...", config.RETENTION_DAYS)
    try:
        deleted = rag.cleanup_old_messages(config.RETENTION_DAYS)
        stats = rag.get_stats()
        logger.info(
            "Cleanup complete: removed %d old messages. %d messages remaining.",
            deleted, stats["total_messages"]
        )
    except Exception as e:
        logger.error("Scheduled cleanup failed: %s", e)


# ── Message Handler ──────────────────────────────────────────────
@app.event("message")
def handle_message(event, say):
    """
    Triggered on every message the bot can see.
    - Indexes ALL messages for cross-channel context
    - Generates drafts only for DMs and @mentions
    """
    user_id = event.get("user", "")

    # Ignore bot messages and message edits/deletions
    if event.get("bot_id") or event.get("subtype"):
        return

    text = event.get("text", "").strip()
    if not text:
        return

    channel_id = event.get("channel", "")
    thread_ts = event.get("thread_ts") or event.get("ts")
    channel_type = event.get("channel_type", "")

    # Index EVERY message for cross-channel intelligence
    index_message(text, user_id, channel_id, event.get("ts", ""))

    # Determine if this message is directed at you:
    # 1. Direct messages (always)
    # 2. @mentions in channels
    is_dm = channel_type in ("im", "mpim")
    is_mention = ("<@" + config.MY_SLACK_USER_ID + ">") in text

    if not is_dm and not is_mention:
        # Already indexed above — no draft needed
        return

    logger.info(
        "New message from %s in %s: %s...",
        get_user_name(user_id),
        get_channel_name(channel_id),
        text[:80]
    )

    # Clean up the message text (remove @mention markup)
    clean_text = re.sub(r"<@\w+>", "", text).strip()

    # Get relevant past conversations from vector store
    # This now uses recency-weighted search across ALL channels
    context_messages = rag.search(
        query=get_user_name(user_id) + " " + clean_text,
        top_k=config.RAG_TOP_K,
    )

    # Get thread context if this is a threaded conversation
    thread_messages = []
    if event.get("thread_ts"):
        thread_messages = get_thread_messages(channel_id, event["thread_ts"])

    # Generate draft using Claude
    draft = drafter.generate_draft(
        incoming_message=clean_text,
        sender_name=get_user_name(user_id),
        channel_name=get_channel_name(channel_id),
        thread_messages=thread_messages,
        context_messages=context_messages,
    )

    if draft:
        send_draft_both(
            channel_id, thread_ts, draft, clean_text,
            sender_name=get_user_name(user_id),
            channel_name=get_channel_name(channel_id),
        )


# ── Button Handlers ──────────────────────────────────────────────
@app.action("send_draft")
def handle_send(ack, action, respond):
    """Send the draft as an actual message, then clean up both copies."""
    ack()
    data = json.loads(action["value"])

    try:
        slack.chat_postMessage(
            channel=data["channel"],
            thread_ts=data["thread_ts"],
            text=data["draft"],
        )
        # Clean up: replace the message the button was on
        respond(text="✅ Reply sent!", replace_original=True)
        # Clean up: delete the DM copy
        delete_dm_message(data.get("dm_channel"), data.get("dm_ts"))
        logger.info("Draft sent as actual message.")
    except SlackApiError as e:
        respond(text="❌ Failed to send: " + str(e))
        logger.error("Failed to send draft: %s", e)


@app.action("regenerate_draft")
def handle_regenerate(ack, action, respond):
    """Generate a new draft for the same message."""
    ack()
    data = json.loads(action["value"])

    sender_name = data.get("sender_name", "(unknown)")
    channel_name = data.get("channel_name") or get_channel_name(data["channel"])

    # Re-search and re-generate
    context_messages = rag.search(query=data["original_text"])
    thread_messages = get_thread_messages(data["channel"], data["thread_ts"])

    draft = drafter.generate_draft(
        incoming_message=data["original_text"],
        sender_name=sender_name,
        channel_name=channel_name,
        thread_messages=thread_messages,
        context_messages=context_messages,
    )

    if draft:
        permalink = get_permalink(data["channel"], data["thread_ts"])
        if permalink:
            header_text = "💬 *" + sender_name + "* in *#" + channel_name + "* (<" + permalink + "|view message>):\n>" + data["original_text"]
        else:
            header_text = "💬 *" + sender_name + "* in *#" + channel_name + "*:\n>" + data["original_text"]

        # Keep the same dm_channel/dm_ts so cleanup still works
        dm_channel = data.get("dm_channel", "")
        dm_ts = data.get("dm_ts", "")

        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": header_text},
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "✏️ *New draft:*\n\n" + draft},
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Send"},
                        "style": "primary",
                        "action_id": "send_draft",
                        "value": json.dumps({
                            "channel": data["channel"],
                            "thread_ts": data["thread_ts"],
                            "draft": draft,
                            "dm_channel": dm_channel,
                            "dm_ts": dm_ts,
                        }),
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🔄 Regenerate"},
                        "action_id": "regenerate_draft",
                        "value": json.dumps({
                            "channel": data["channel"],
                            "thread_ts": data["thread_ts"],
                            "original_text": data["original_text"],
                            "sender_name": sender_name,
                            "channel_name": channel_name,
                            "dm_channel": dm_channel,
                            "dm_ts": dm_ts,
                        }),
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Dismiss"},
                        "action_id": "dismiss_draft",
                        "value": json.dumps({
                            "dm_channel": dm_channel,
                            "dm_ts": dm_ts,
                        }),
                    },
                ],
            },
        ]
        respond(blocks=blocks, replace_original=True)
    else:
        respond(text="Couldn't generate a new draft. Try again?")


@app.action("dismiss_draft")
def handle_dismiss(ack, action, respond):
    """Dismiss the draft and clean up the DM copy."""
    ack()
    data = json.loads(action.get("value", "{}"))
    # Clean up: replace the message the button was on
    respond(delete_original=True)
    # Clean up: delete the DM copy
    delete_dm_message(data.get("dm_channel"), data.get("dm_ts"))


# ── Slash Command: Stats ─────────────────────────────────────────
@app.command("/draft-stats")
def handle_stats(ack, respond):
    """Show stats about the vector store and retention."""
    ack()
    stats = rag.get_stats()

    text_parts = [
        "📊 *Draft Agent Stats*",
        "• Messages indexed: " + str(stats["total_messages"]),
        "• Retention window: " + str(stats.get("retention_days", "?")) + " days",
        "• Recency weight: " + str(stats.get("recency_weight", "?")),
        "• Storage: " + stats["storage_path"],
    ]
    if stats.get("oldest_message"):
        text_parts.append("• Oldest message: " + stats["oldest_message"])
    if stats.get("newest_message"):
        text_parts.append("• Newest message: " + stats["newest_message"])

    respond(text="\n".join(text_parts))


# ── Start the App ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Slack Draft Agent - Starting...")
    print("  Real-time indexing | 60-day rolling window | Dual delivery")
    print("=" * 60)

    stats = rag.get_stats()
    if stats["total_messages"] == 0:
        print(
            "\n⚠️  Vector store is empty!"
            "\n   Run 'python ingest.py' first to load your Slack history."
            "\n   The agent will still work, but drafts won't have context.\n"
        )
    else:
        print("\n✅ " + str(stats["total_messages"]) + " messages in vector store.")
        if stats.get("oldest_message"):
            print("   Date range: " + stats["oldest_message"] + " to " + stats.get("newest_message", "now"))

    print("👤 Watching messages for user: " + config.MY_SLACK_USER_ID)
    print("🤖 Using model: " + config.CLAUDE_MODEL)
    print("📦 Retention: " + str(config.RETENTION_DAYS) + " days | Recency weight: " + str(config.RECENCY_WEIGHT))

    # Start background scheduler for daily cleanup
    scheduler = BackgroundScheduler()
    scheduler.add_job(daily_cleanup, "interval", hours=24)
    scheduler.start()
    print("🧹 Daily cleanup scheduled (every 24 hours)")

    # Run one cleanup on startup
    daily_cleanup()

    print("📡 Connecting to Slack via Socket Mode...\n")

    # ── Health Check Server (for Render.com) ──────────────────
    flask_app = Flask(__name__)

    @flask_app.route('/')
    def health_check():
        return "Slack Draft Agent is running!", 200

    def run_flask():
        # Render provides a PORT environment variable
        port = int(os.environ.get("PORT", 8080))
        flask_app.run(host='0.0.0.0', port=port)

    # Start Flask in a background thread
    Thread(target=run_flask, daemon=True).start()
    print(f"🚀 Health check server started (port {os.environ.get('PORT', 8080)})")

    handler = SocketModeHandler(app, config.SLACK_APP_TOKEN)
    handler.start()
