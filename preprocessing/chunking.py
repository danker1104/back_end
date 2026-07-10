from typing import Any, Dict, List, Optional

from config import CHUNK_OVERLAP, CHUNK_SIZE
from preprocessing.load_documents import prepare_documents


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """텍스트를 고정 크기 청크로 나눈다."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def create_document_chunks(
    documents: Optional[List[Dict[str, Any]]] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """정제된 문서 리스트를 청크 리스트로 변환한다. 메타데이터는 유지한다."""
    source_documents = documents if documents is not None else prepare_documents()
    chunks: List[Dict[str, Any]] = []

    for document in source_documents:
        text = document.get("text", "")
        if not text:
            continue

        metadata = {
            key: value for key, value in document.items() if key != "text"
        }
        subchunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for index, chunk_text in enumerate(subchunks):
            chunk_payload = {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": index,
                },
            }
            chunks.append(chunk_payload)

    return chunks


if __name__ == "__main__":
    chunks = create_document_chunks()
    print(f"Created {len(chunks)} chunks")
    if chunks:
        print(chunks[0])
