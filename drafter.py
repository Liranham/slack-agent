"""
Draft Generator - uses Claude to write draft responses.

Takes an incoming Slack message + relevant context from the vector store
and produces a natural-sounding draft reply.

Now with cross-channel intelligence: context may come from ANY channel
the bot monitors, not just the current conversation.
"""
import logging
from typing import Optional

import anthropic
import config

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy-init Anthropic client."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """\
You are a smart draft assistant for a business professional on Slack.

Your job: write a draft reply to an incoming Slack message, matching the user's
natural communication style. The draft should be ready to send with minimal edits.

CONTEXT INTELLIGENCE:
- You receive context from across ALL Slack channels the user participates in
- Context messages are labeled with their channel and timestamp
- Use cross-channel context to give more informed, relevant replies
- If someone asks about a topic that was discussed in another channel, use that info
- Pay special attention to RECENT messages — they indicate current priorities and conversations
- Context from private DMs or channels may contain sensitive info — reference the knowledge
  but don't explicitly reveal which private conversation it came from

RULES:
- Match the tone of the conversation (casual for casual, professional for professional)
- Be concise — Slack messages should be short and direct
- Use context from past conversations to give informed, relevant replies
- If the context shows a pattern of how the user replies, mimic that style
- Don't use overly formal language unless the conversation calls for it
- Don't add greetings like "Hi!" unless the incoming message has one
- If you're unsure about specific facts, acknowledge it briefly
- Write ONLY the draft reply text — no explanations, no "Here's a draft:" prefix
- If the message doesn't need a reply (e.g. just a reaction or "thanks"), respond with: [NO_REPLY_NEEDED]
"""


def generate_draft(
    incoming_message,
    sender_name,
    channel_name,
    thread_messages=None,
    context_messages=None,
):
    # type: (str, str, str, Optional[list], Optional[list]) -> Optional[str]
    """
    Generate a draft reply to an incoming Slack message.

    Args:
        incoming_message: The message that was sent to the user
        sender_name: Who sent it
        channel_name: Which channel or DM
        thread_messages: Recent messages in the same thread (if any)
        context_messages: Relevant past messages from RAG search (cross-channel)

    Returns:
        Draft reply text, or None if no reply is needed
    """
    # Build the context section
    context_parts = []

    if context_messages:
        context_parts.append("=== RELEVANT CONTEXT (from across all channels) ===")
        for msg in context_messages:
            doc = msg.get("document", "")
            score = msg.get("score", 0)
            recency = msg.get("recency", 0)
            # Add relevance indicator for Claude
            if score > 0.7:
                doc = "[HIGH RELEVANCE] " + doc
            elif score > 0.5:
                doc = "[MODERATE RELEVANCE] " + doc
            context_parts.append(doc)
        context_parts.append("")

    if thread_messages:
        context_parts.append("=== CURRENT THREAD (recent messages) ===")
        for msg in thread_messages:
            name = msg.get("user_name", "Unknown")
            text = msg.get("text", "")
            context_parts.append(name + ": " + text)
        context_parts.append("")

    context_text = "\n".join(context_parts) if context_parts else "(No prior context available)"

    # Build the user prompt
    user_prompt = (
        context_text + "\n\n"
        "=== NEW INCOMING MESSAGE ===\n"
        "From: " + sender_name + "\n"
        "Channel: #" + channel_name + "\n"
        "Message: " + incoming_message + "\n\n"
        "Write a draft reply:"
    )

    logger.info(
        "Generating draft for message from %s in #%s",
        sender_name, channel_name
    )
    logger.debug("Prompt length: %d chars", len(user_prompt))

    try:
        response = _get_client().messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        draft = response.content[0].text.strip()

        # Check if Claude thinks no reply is needed
        if "[NO_REPLY_NEEDED]" in draft:
            logger.info("Claude determined no reply is needed.")
            return None

        logger.info("Draft generated (%d chars)", len(draft))
        return draft

    except Exception as e:
        logger.error("Failed to generate draft: %s", e)
        return None
