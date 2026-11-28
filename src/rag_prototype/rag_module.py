"""RAG 파이프라인 8단계: 로드 → 분할 → 임베딩 → 벡터 DB → 검색 → 프롬프트 → LLM → 출력.

LangChain 기본 RAG 예제 구조를 바탕으로, 답변 모델을 Claude로 바꾸고
임베딩을 로컬 다국어 모델(bge-m3)로 바꿨다. 실험 파라미터는 .env에서 읽는다.
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sentence_transformers import CrossEncoder
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
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3")     # prompts.py의 키

# 리랭커: 후보를 넉넉히(RERANK_FETCH_K개) 뽑은 뒤, 질문과 조각을 함께 읽는 모델로 다시 순위를 매겨 TOP_K개만 쓴다
RERANK = os.getenv("RERANK", "false").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_FETCH_K = int(os.getenv("RERANK_FETCH_K", "20"))


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


def format_docs(docs):
    """검색된 조각을 "[N쪽] 내용" 형식으로 이어 붙인다 (출처 쪽수를 LLM이 볼 수 있게)."""
    return "\n\n".join(
        f"[{d.metadata.get('page', 0) + 1}쪽] {d.page_content}" for d in docs
    )


def create_rag_chain(pdf_path):
    # [1단계] 문서 로드
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # [2단계] 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_documents = text_splitter.split_documents(docs)

    # [3~4단계] 임베딩 및 벡터 DB 저장
    # Claude는 임베딩 모델이 없어서, 내 컴퓨터에서 도는 무료 임베딩 모델을 쓴다
    # (처음 실행할 때 모델을 내려받느라 몇 분 걸린다)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # [5단계] 검색기
    if RERANK:
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
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
