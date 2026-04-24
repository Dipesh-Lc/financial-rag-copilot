from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from app.embeddings.hf_embeddings import get_embeddings
from app.logging_config import get_logger

logger = get_logger(__name__)


def get_vector_store(
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_dir: Path = CHROMA_PERSIST_DIR,
) -> Chroma:
    embeddings = get_embeddings()
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    logger.info("Chroma collection '%s' ready at %s", collection_name, persist_dir)
    return store


def add_chunks(
    chunks: list[dict],
    store: Chroma | None = None,
    batch_size: int = 128,
) -> Chroma:
    if store is None:
        store = get_vector_store()

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={k: v for k, v in chunk.items() if k != "text"},
        )
        for chunk in chunks
    ]
    ids = [chunk["chunk_id"] for chunk in chunks]

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        store.add_documents(documents=documents[start:end], ids=ids[start:end])
        logger.info(
            "Indexed chunks %d-%d of %d",
            start + 1,
            min(end, len(documents)),
            len(documents),
        )

    logger.info("Upserted %d chunks into Chroma", len(chunks))
    return store


def collection_stats(store: Chroma) -> dict:
    return {"total_chunks": store._collection.count()}
