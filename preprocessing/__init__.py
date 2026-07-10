"""Step 2 preprocessing utilities for legal JSON documents."""

from .chunking import create_document_chunks, split_text_into_chunks
from .load_documents import (
    clean_document,
    discover_json_files,
    extract_metadata,
    extract_text_from_document,
    load_json_documents,
    normalize_text,
    prepare_documents,
)

__all__ = [
    "clean_document",
    "create_document_chunks",
    "discover_json_files",
    "extract_metadata",
    "extract_text_from_document",
    "load_json_documents",
    "normalize_text",
    "prepare_documents",
    "split_text_into_chunks",
]
