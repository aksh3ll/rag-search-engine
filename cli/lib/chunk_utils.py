import re
from typing import List


def chunk_words(text: str, chunk_size: int = 200, overlap: int = 0) -> List[List[str]]:
    text = text.strip()
    if not text:
        return []
    words = text.split()
    return [
        words[i - overlap if i - overlap >= 0 else 0 :i + chunk_size]
        for i in range(0, len(words), chunk_size)
    ]

def chunk_sentences(text: str, max_chunk_size: int = 4, overlap: int = 0) -> List[List[str]]:
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks = []
    i = 0
    while i < len(sentences):
        chunks.append(sentences[i: i + max_chunk_size])
        if i + max_chunk_size >= len(sentences):
            break
        i += max_chunk_size - overlap
    return chunks
