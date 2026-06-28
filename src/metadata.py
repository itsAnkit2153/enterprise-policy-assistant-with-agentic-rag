from datetime import datetime, timezone
import hashlib


def enrich_metadata(chunks: list, upload_time: str | None = None) -> list:
    """
    Enrich each chunk with the full metadata schema from prompt 2:
      - source_file      : original uploaded filename
      - document_type    : always "policy"
      - upload_time      : ISO-8601 UTC timestamp of this indexing run
      - chunk_id         : stable unique ID (hash of source + page + position)
      - document_label   : human-readable title derived from filename
      - chunk_index      : sequential position across all chunks
    """
    ts = upload_time or datetime.now(timezone.utc).isoformat()

    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source_filename", "unknown")
        page   = chunk.metadata.get("page", 0)

        # Stable, reproducible ID
        uid = hashlib.md5(f"{source}::{page}::{idx}".encode()).hexdigest()[:12]

        chunk.metadata["chunk_id"]      = f"chunk_{uid}"
        chunk.metadata["chunk_index"]   = idx
        chunk.metadata["source_file"]   = source          # prompt-2 field name
        chunk.metadata["document_type"] = "policy"
        chunk.metadata["upload_time"]   = ts

        # Human-readable label
        base = source.rsplit(".", 1)[0] if "." in source else source
        chunk.metadata["document_label"] = (
            base.replace("_", " ").replace("-", " ").title()
        )

    return chunks
