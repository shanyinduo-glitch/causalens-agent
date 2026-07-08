from __future__ import annotations

from collections import Counter
from pathlib import Path

from causalens.rag.chunker import create_chunks_from_documents
from causalens.rag.document_loader import load_markdown_documents


def main() -> None:
    knowledge_dir = Path("knowledge")

    documents = load_markdown_documents(knowledge_dir)

    chunks = create_chunks_from_documents(
        documents=documents,
        chunk_size=500,
        chunk_overlap=80,
    )

    print("=" * 60)
    print("Knowledge Base Inspection")
    print("=" * 60)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    print("\nDocuments:")
    for document in documents:
        content_length = len(document["content"])
        print(f"- {document['source']} ({content_length} characters)")

    print("\nChunks per source:")
    source_counts = Counter(chunk["source"] for chunk in chunks)
    for source, count in sorted(source_counts.items()):
        print(f"- {source}: {count} chunks")

    print("\nFirst 5 chunks:")
    for chunk in chunks[:5]:
        preview = chunk["content"].replace("\n", " ")
        preview = preview[:120] + "..." if len(preview) > 120 else preview

        print("-" * 60)
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"source:   {chunk['source']}")
        print(f"index:    {chunk['chunk_index']}")
        print(f"preview:  {preview}")


if __name__ == "__main__":
    main()