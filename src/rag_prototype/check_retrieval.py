"""검색 진단: 질문에 대해 어떤 조각이 몇 등으로 검색되는지 확인한다.

LLM을 부르지 않으므로 API 비용이 들지 않는다.
rag_module.py와 같은 설정(.env의 CHUNK_SIZE, CHUNK_OVERLAP, SPLIT_MODE, EMBEDDING_MODEL)으로 색인한다.

실행 (app.py가 있는 폴더에서):
    python check_retrieval.py 문서.pdf "이 사업의 배경과 목적이 뭐야?"
    python check_retrieval.py 문서.pdf "이 사업의 배경과 목적이 뭐야?" --k 10 --find "사업 배경 및 목적"
    python check_retrieval.py 문서.pdf "보안 요구사항을 모두 알려줘" --find "TLS" --rerank

--find: 정답 조각에 들어 있을 문구. 그 조각이 전체에서 몇 등인지 알려 준다.
--rerank: 임베딩 상위 RERANK_FETCH_K개를 리랭커로 다시 정렬해, 리랭킹 전후 순위를 비교한다.
"""
import argparse

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag_module import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RERANK_FETCH_K,
    RERANK_MODEL,
    SPLIT_MODE,
    load_and_split,
    page_label,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=6, help="출력할 상위 조각 수")
    parser.add_argument("--find", default=None, help="정답 조각에 들어 있을 문구")
    parser.add_argument("--rerank", action="store_true", help="리랭킹 전후 순위 비교")
    args = parser.parse_args()

    chunks = load_and_split(args.pdf)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = FAISS.from_documents(chunks, embeddings)

    print(f"설정: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, 분할={SPLIT_MODE}, 임베딩={EMBEDDING_MODEL}")
    print(f"전체 조각 수: {len(chunks)}")

    # 전체 조각을 거리순으로 받아 순위를 매긴다 (FAISS 점수는 거리라서 작을수록 비슷함)
    ranked = store.similarity_search_with_score(args.question, k=len(chunks))

    print(f"\n질문: {args.question}\n")
    for rank, (doc, score) in enumerate(ranked[: args.k], start=1):
        page = page_label(doc)
        preview = doc.page_content.replace("\n", " ")[:120]
        print(f"[{rank}등] {page}쪽 · 거리 {score:.3f} · {len(doc.page_content)}자")
        print(f"      {preview}...")

    if args.find:
        hits = [(r, d) for r, (d, _) in enumerate(ranked, start=1) if args.find in d.page_content]
        print()
        if not hits:
            print(f"'{args.find}'가 들어 있는 조각이 없습니다. (조각 경계에서 잘렸을 수 있음)")
        for r, d in hits:
            page = page_label(d)
            print(f"'{args.find}' 포함 조각: {r}등 / 전체 {len(ranked)}개 ({page}쪽)")

    if args.rerank:
        from sentence_transformers import CrossEncoder

        candidates = [d for d, _ in ranked[:RERANK_FETCH_K]]
        model = CrossEncoder(RERANK_MODEL)
        scores = model.predict([(args.question, d.page_content) for d in candidates])
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        print(f"\n[리랭킹] 임베딩 상위 {RERANK_FETCH_K}개를 {RERANK_MODEL}로 다시 정렬\n")
        for rank, (doc, score) in enumerate(reranked[: args.k], start=1):
            page = page_label(doc)
            before = candidates.index(doc) + 1
            preview = doc.page_content.replace("\n", " ")[:120]
            print(f"[{rank}등] {page}쪽 · 점수 {score:.3f} · 리랭킹 전 {before}등")
            print(f"      {preview}...")

        if args.find:
            print()
            hits = [r for r, (d, _) in enumerate(reranked, start=1) if args.find in d.page_content]
            if not hits:
                print(f"'{args.find}' 포함 조각이 임베딩 상위 {RERANK_FETCH_K}개 안에 없어 리랭킹 대상이 아닙니다.")
            for r in hits:
                print(f"리랭킹 후 '{args.find}' 포함 조각: {r}등 / 후보 {len(reranked)}개")


if __name__ == "__main__":
    main()
