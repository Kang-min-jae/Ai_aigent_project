"""RAG 파이프라인 8단계: 로드 → 분할 → 임베딩 → 벡터 DB → 검색 → 프롬프트 → LLM → 출력.

LangChain 기본 RAG 예제 구조를 바탕으로, 답변 모델을 Claude로 바꾸고
임베딩을 로컬 다국어 모델(bge-m3)로 바꿨다. 실험 파라미터는 .env에서 읽는다.
"""
import os
import re
from bisect import bisect_right

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from prompts import PROMPTS

# .env 값을 윈도우 환경변수보다 우선 적용한다 (예전 키가 남아 있어도 무시)
load_dotenv(override=True)

# 도메인·실험에 따라 바뀌는 값은 .env에서 읽는다 (하드코딩하지 않음)
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-haiku-5-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "3"))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")  # similarity 또는 mmr
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3.1")     # prompts.py의 키
# merged: 쪽을 이어 붙인 뒤 분할 (쪽 경계에서 문장이 잘리지 않음) / page: 쪽마다 따로 분할 (원래 예제)
SPLIT_MODE = os.getenv("SPLIT_MODE", "merged")

# 리랭커: 후보를 넉넉히(RERANK_FETCH_K개) 뽑은 뒤, 질문과 조각을 함께 읽는 모델로 다시 순위를 매겨 TOP_K개만 쓴다
RERANK = os.getenv("RERANK", "false").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_FETCH_K = int(os.getenv("RERANK_FETCH_K", "20"))

# 쪽 번호 꼬리말("- 2 -" 같은 줄). 쪽을 이어 붙일 때 문장 중간에 끼지 않도록 지운다
PAGE_FOOTER = re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE)


def load_and_split(pdf_path):
    """PDF를 읽어 조각으로 나눈다. 각 조각에는 쪽 정보(metadata["page_label"])를 붙인다."""
    pages = PyMuPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    if SPLIT_MODE == "page":
        # 원래 예제 방식: 쪽마다 따로 자르므로 쪽을 넘어가는 문장·목록이 두 조각으로 갈린다
        chunks = splitter.split_documents(pages)
        for c in chunks:
            c.metadata["page_label"] = str(c.metadata.get("page", 0) + 1)
        return chunks

    # 쪽 꼬리말을 지우고 전체를 한 덩어리로 이어 붙인 뒤 자른다.
    # 각 쪽이 전체 텍스트의 몇 번째 글자에서 시작하는지 기록해 두었다가, 조각의 쪽 범위를 계산한다.
    texts, starts, pos = [], [], 0
    for p in pages:
        text = PAGE_FOOTER.sub("", p.page_content).strip()
        starts.append(pos)
        texts.append(text)
        pos += len(text) + 1  # 이어 붙일 때 넣는 줄바꿈 1글자
    full_text = "\n".join(texts)

    chunks = splitter.create_documents([full_text], metadatas=[{"source": pdf_path}])
    for c in chunks:
        begin = c.metadata["start_index"]
        end = begin + max(len(c.page_content) - 1, 0)
        first = bisect_right(starts, begin) - 1
        last = bisect_right(starts, end) - 1
        c.metadata["page"] = first  # 0부터 시작 (PyMuPDFLoader와 같은 규칙)
        c.metadata["page_label"] = f"{first + 1}" if first == last else f"{first + 1}~{last + 1}"
    return chunks


def rerank_docs(question, docs, model, top_k):
    """질문-조각 쌍의 관련도 점수를 매겨 높은 순으로 top_k개를 돌려준다.

    임베딩 검색은 질문과 조각을 따로 벡터로 바꿔 비교하지만,
    리랭커(크로스 인코더)는 둘을 한 번에 읽어 관련도를 판단하므로 더 정밀하다.
    """
    if not docs:
        return []
    scores = model.predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


def page_label(doc):
    return doc.metadata.get("page_label") or str(doc.metadata.get("page", 0) + 1)


def format_docs(docs):
    """검색된 조각을 "[N쪽] 내용" 형식으로 이어 붙인다 (출처 쪽수를 LLM이 볼 수 있게)."""
    return "\n\n".join(f"[{page_label(d)}쪽] {d.page_content}" for d in docs)


def create_rag_chain(pdf_path):
    """질문을 받아 {"question", "docs", "answer"}를 돌려주는 체인을 만든다.

    docs(검색된 조각)도 함께 돌려줘서 화면에서 근거를 보여 줄 수 있게 한다.
    """
    # [1~2단계] 문서 로드 및 분할
    split_documents = load_and_split(pdf_path)

    # [3~4단계] 임베딩 및 벡터 DB 저장
    # Claude는 임베딩 모델이 없어서, 내 컴퓨터에서 도는 무료 임베딩 모델을 쓴다
    # (처음 실행할 때 모델을 내려받느라 몇 분 걸린다)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # [5단계] 검색기
    if RERANK:
        from sentence_transformers import CrossEncoder

        # 1차: 임베딩으로 후보를 넉넉히 뽑는다 → 2차: 리랭커로 TOP_K개를 고른다
        base = vectorstore.as_retriever(
            search_type=SEARCH_TYPE,
            search_kwargs={"k": RERANK_FETCH_K},
        )
        reranker = CrossEncoder(RERANK_MODEL)
        retriever = RunnableLambda(
            lambda q: rerank_docs(q, base.invoke(q), reranker, TOP_K)
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type=SEARCH_TYPE,
            search_kwargs={"k": TOP_K},
        )

    # [6~7단계] 프롬프트 및 LLM (Claude)
    # 프롬프트는 prompts.py에 버전별로 두고 .env의 PROMPT_VERSION으로 고른다
    prompt = ChatPromptTemplate.from_template(PROMPTS[PROMPT_VERSION])
    # Claude 5.5 모델은 temperature 설정을 받지 않는다 (넣으면 400 오류)
    llm = ChatAnthropic(model=ANSWER_MODEL, max_tokens=2000)

    # [8단계] 체인 생성
    # 질문 → 검색(docs) → 답변 생성. 검색 결과를 답변과 함께 돌려준다
    answer_chain = (
        RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain = (
        RunnableLambda(lambda q: {"question": q, "docs": retriever.invoke(q)})
        | RunnablePassthrough.assign(answer=answer_chain)
    )

    return rag_chain
