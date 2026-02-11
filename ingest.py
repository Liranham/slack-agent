"""
Slack History Ingestion Script.

Run this ONCE to pull your entire Slack history into the vector database.
After this initial ingestion, the main app.py handles new messages automatically.

Usage:
    python ingest.py
"""
import logging
import sys
import time
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import config
import rag

# ── Setup ────────────────────────────────────────────────────────
config.validate()
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

slack = WebClient(token=config.SLACK_BOT_TOKEN)

# ── User Cache ───────────────────────────────────────────────────
_user_cache = {}


def get_user_name(user_id: str) -> str:
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


# ── Conversation Discovery ───────────────────────────────────────
def get_all_conversations() -> list[dict]:
    """Get all channels, DMs, and group DMs the bot has access to."""
    conversations = []
    cursor = None

    # Types: public_channel, private_channel, mpim (group DM), im (direct message)
    types = "public_channel,private_channel,mpim,im"

    while True:
        try:
            result = slack.conversations_list(
                types=types,
                limit=200,
                cursor=cursor,
            )
            conversations.extend(result["channels"])

            cursor = result.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

            time.sleep(1)  # respect rate limits

        except SlackApiError as e:
            logger.error(f"Error listing conversations: {e}")
            break

    logger.info(f"Found {len(conversations)} conversations")
    return conversations


# ── Message Fetching ─────────────────────────────────────────────
def fetch_channel_history(channel_id: str, channel_name: str) -> list[dict]:
    """Pull all messages from a single channel/DM."""
    messages = []
    cursor = None
    page = 0

    while True:
        try:
            result = slack.conversations_history(
                channel=channel_id,
                limit=200,  # max per request
                cursor=cursor,
            )

            for msg in result.get("messages", []):
                # Skip bot messages and system messages
                if msg.get("subtype") in ("bot_message", "channel_join", "channel_leave"):
                    continue
                if not msg.get("text"):
                    continue

                user_id = msg.get("user", "unknown")
                messages.append({
                    "id": f"{channel_id}_{msg['ts']}",
                    "text": msg["text"],
                    "user_id": user_id,
                    "user_name": get_user_name(user_id),
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "timestamp": msg["ts"],
                })

            cursor = result.get("response_metadata", {}).get("next_cursor")
            page += 1

            if not cursor:
                break

            # Respect rate limits (important!)
            time.sleep(1.2)

        except SlackApiError as e:
            if e.response["error"] == "not_in_channel":
                logger.warning(f"Bot not in #{channel_name} — skipping")
                break
            elif e.response["error"] == "ratelimited":
                retry_after = int(e.response.headers.get("Retry-After", 30))
                logger.warning(f"Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            else:
                logger.error(f"Error fetching #{channel_name}: {e}")
                break

    return messages


# ── Main Ingestion ───────────────────────────────────────────────
def run_ingestion():
    """Main ingestion process."""
    print("\n" + "=" * 60)
    print("  Slack History Ingestion")
    print("=" * 60)

    # Step 1: Get all conversations
    print("\n[1/3] Discovering conversations...")
    conversations = get_all_conversations()

    # Step 2: Fetch messages from each
    print(f"\n[2/3] Fetching messages from {len(conversations)} conversations...")
    total_messages = 0
    batch = []
    batch_size = 100  # process in batches to manage memory

    for i, conv in enumerate(conversations):
        channel_id = conv["id"]
        channel_name = conv.get("name") or conv.get("user", "dm")

        # For DMs, try to get the other person's name
        if conv.get("is_im"):
            other_user = conv.get("user", "")
            channel_name = f"DM-{get_user_name(other_user)}"

        print(f"  [{i+1}/{len(conversations)}] #{channel_name}...", end=" ", flush=True)

        messages = fetch_channel_history(channel_id, channel_name)
        print(f"{len(messages)} messages")

        batch.extend(messages)
        total_messages += len(messages)

        # Store in batches
        if len(batch) >= batch_size:
            rag.add_messages(batch)
            batch = []

    # Store remaining messages
    if batch:
        rag.add_messages(batch)

    # Step 3: Summary
    stats = rag.get_stats()
    print(f"\n[3/3] Done!")
    print(f"\n{'=' * 60}")
    print(f"  INGESTION COMPLETE")
    print(f"  Total messages ingested: {total_messages}")
    print(f"  Total in vector store:   {stats['total_messages']}")
    print(f"  Storage location:        {stats['storage_path']}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    try:
        run_ingestion()
    except EnvironmentError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nIngestion cancelled.")
        sys.exit(0)
