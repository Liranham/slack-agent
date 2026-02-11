"""
RAG (Retrieval-Augmented Generation) module.

Handles:
- Storing Slack messages as vector embeddings in ChromaDB
- Searching for relevant past messages with recency boosting
- 60-day rolling window cleanup of old messages
"""
import logging
import time
from datetime import datetime, timedelta

import chromadb
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# ── Globals (initialized on first use) ──────────────────────────
_chroma_client = None
_collection = None
_openai_client = None

# How many days to keep messages (configurable via .env)
RETENTION_DAYS = int(getattr(config, "RETENTION_DAYS", 60))

# Recency boost: how much to favor recent messages in search
# 0.0 = no boost, 1.0 = heavy recency preference
RECENCY_WEIGHT = float(getattr(config, "RECENCY_WEIGHT", 0.3))


def _get_openai():
    """Lazy-init OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _get_collection():
    """Lazy-init ChromaDB collection."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name="slack_messages",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB initialized at {config.CHROMA_PATH} "
            f"({_collection.count()} documents)"
        )
    return _collection


def embed_texts(texts):
    """Generate embeddings for a list of texts using OpenAI."""
    if not texts:
        return []
    response = _get_openai().embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def add_messages(messages):
    """
    Store Slack messages in the vector database.

    Each message dict should have:
        - id: unique message ID
        - text: the message content
        - user_id: who sent it
        - user_name: display name of sender
        - channel_id: which channel/DM
        - channel_name: display name of channel
        - timestamp: Slack ts value
    """
    if not messages:
        return

    collection = _get_collection()

    ids = []
    documents = []
    metadatas = []

    for msg in messages:
        if not msg.get("text", "").strip():
            continue

        msg_id = msg["id"]
        text = msg["text"]

        ts = float(msg.get("timestamp", 0))
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "unknown"

        document = (
            f"[{date_str}] {msg.get('user_name', 'Unknown')} "
            f"in #{msg.get('channel_name', 'unknown')}: {text}"
        )

        ids.append(msg_id)
        documents.append(document)
        metadatas.append({
            "user_id": msg.get("user_id", ""),
            "user_name": msg.get("user_name", "Unknown"),
            "channel_id": msg.get("channel_id", ""),
            "channel_name": msg.get("channel_name", "unknown"),
            "timestamp": str(msg.get("timestamp", "")),
            "raw_text": text[:500],
            "indexed_at": str(time.time()),
        })

    if not documents:
        return

    # Embed in batches of 100 to avoid API limits
    all_embeddings = []
    for i in range(0, len(documents), 100):
        batch = documents[i:i + 100]
        all_embeddings.extend(embed_texts(batch))

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )
    logger.info(f"Stored {len(documents)} messages. Total: {collection.count()}")


def _recency_score(timestamp_str, max_age_days=60):
    """
    Calculate a recency score between 0.0 and 1.0.
    Recent messages score higher.
    """
    try:
        ts = float(timestamp_str)
        age_seconds = time.time() - ts
        age_days = age_seconds / 86400
        if age_days <= 0:
            return 1.0
        if age_days >= max_age_days:
            return 0.0
        return 1.0 - (age_days / max_age_days)
    except (ValueError, TypeError):
        return 0.5  # default for unknown timestamps


def search(query, top_k=None, filter_channel=None):
    """
    Search for messages most relevant to the query.
    Combines semantic similarity with recency boosting.

    Returns a list of dicts with: document, metadata, distance, score
    """
    collection = _get_collection()
    if collection.count() == 0:
        logger.warning("Vector store is empty.")
        return []

    k = top_k or config.RAG_TOP_K

    # Fetch more results than needed so we can re-rank with recency
    fetch_k = min(k * 3, collection.count())

    where = None
    if filter_channel:
        where = {"channel_id": filter_channel}

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Combine semantic similarity with recency score
    candidates = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]

        # Cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 = identical, 0 = opposite
        similarity = max(0, 1.0 - distance)

        # Recency boost
        recency = _recency_score(metadata.get("timestamp", ""), RETENTION_DAYS)

        # Combined score: weighted blend of similarity and recency
        combined = (1 - RECENCY_WEIGHT) * similarity + RECENCY_WEIGHT * recency

        candidates.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": metadata,
            "distance": distance,
            "similarity": similarity,
            "recency": recency,
            "score": combined,
        })

    # Sort by combined score (highest first) and take top_k
    candidates.sort(key=lambda x: x["score"], reverse=True)
    output = candidates[:k]

    logger.info(
        f"Search returned {len(output)} results for: {query[:80]}... "
        f"(fetched {fetch_k}, re-ranked with recency)"
    )
    return output


def cleanup_old_messages(days=None):
    """
    Remove messages older than the retention period.
    Returns the number of messages deleted.
    """
    retention = days or RETENTION_DAYS
    collection = _get_collection()
    total = collection.count()

    if total == 0:
        logger.info("No messages to clean up.")
        return 0

    # Calculate the cutoff timestamp
    cutoff = time.time() - (retention * 86400)
    cutoff_str = str(cutoff)

    # ChromaDB where filter: find messages with timestamp < cutoff
    # We need to get all messages and filter manually since ChromaDB
    # doesn't support < comparison on string timestamps easily
    try:
        # Get all message IDs and their timestamps
        all_data = collection.get(
            include=["metadatas"],
            limit=total,
        )

        ids_to_delete = []
        for i, metadata in enumerate(all_data["metadatas"]):
            try:
                ts = float(metadata.get("timestamp", "0"))
                if ts > 0 and ts < cutoff:
                    ids_to_delete.append(all_data["ids"][i])
            except (ValueError, TypeError):
                continue

        if ids_to_delete:
            # Delete in batches of 500
            for i in range(0, len(ids_to_delete), 500):
                batch = ids_to_delete[i:i + 500]
                collection.delete(ids=batch)

            logger.info(
                f"Cleanup: removed {len(ids_to_delete)} messages older than "
                f"{retention} days. Remaining: {collection.count()}"
            )
        else:
            logger.info(f"Cleanup: no messages older than {retention} days found.")

        return len(ids_to_delete)

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 0


def get_stats():
    """Return stats about the vector store."""
    collection = _get_collection()
    total = collection.count()

    # Try to get age range of stored messages
    oldest_ts = None
    newest_ts = None
    if total > 0:
        try:
            all_data = collection.get(include=["metadatas"], limit=total)
            timestamps = []
            for m in all_data["metadatas"]:
                try:
                    ts = float(m.get("timestamp", "0"))
                    if ts > 0:
                        timestamps.append(ts)
                except (ValueError, TypeError):
                    pass
            if timestamps:
                oldest_ts = min(timestamps)
                newest_ts = max(timestamps)
        except Exception:
            pass

    stats = {
        "total_messages": total,
        "storage_path": config.CHROMA_PATH,
        "retention_days": RETENTION_DAYS,
        "recency_weight": RECENCY_WEIGHT,
    }
    if oldest_ts:
        stats["oldest_message"] = datetime.fromtimestamp(oldest_ts).strftime("%Y-%m-%d")
    if newest_ts:
        stats["newest_message"] = datetime.fromtimestamp(newest_ts).strftime("%Y-%m-%d")

    return stats
