from __future__ import annotations

from causalens.rag.rag_tool import retrieve_docs


QUESTIONS = [
    "passed 字段是什么意思？",
    "study_hours 的单位是什么？",
    "这份数据可以用于因果分析吗？",
    "score 的取值范围是什么？",
]


def main() -> None:
    for question in QUESTIONS:
        print("=" * 80)
        print(f"Question: {question}")

        tool_result = retrieve_docs(
            query=question,
            top_k=3,
        )

        print(f"Success: {tool_result.success}")

        if not tool_result.success:
            print(f"Error: {tool_result.error}")
            continue

        results = tool_result.result["results"]

        for index, item in enumerate(results, start=1):
            content_preview = item["content"].replace("\n", " ")
            content_preview = (
                content_preview[:300] + "..."
                if len(content_preview) > 300
                else content_preview
            )

            print("-" * 80)
            print(f"Rank: {index}")
            print(f"Source: {item['source']}")
            print(f"Section: {item.get('section', '')}")
            print(f"Chunk ID: {item['chunk_id']}")
            print(f"Chunk Index: {item['chunk_index']}")
            print(f"Distance: {item['distance']}")
            print(f"Content Preview: {content_preview}")


if __name__ == "__main__":
    main()