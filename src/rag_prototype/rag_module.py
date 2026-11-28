"""RAG 파이프라인 8단계: 로드 → 분할 → 임베딩 → 벡터 DB → 검색 → 프롬프트 → LLM → 출력.

LangChain 기본 RAG 예제 구조를 바탕으로, 답변 모델을 Claude로 바꾸고
임베딩을 로컬 한국어 모델로 바꿨다. 실험 파라미터는 .env에서 읽는다.
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# .env 값을 윈도우 환경변수보다 우선 적용한다 (예전 키가 남아 있어도 무시)
load_dotenv(override=True)

# 도메인·실험에 따라 바뀌는 값은 .env에서 읽는다 (하드코딩하지 않음)
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-haiku-5-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "3"))


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
    # Claude는 임베딩 모델이 없어서, 내 컴퓨터에서 도는 무료 한국어 임베딩 모델을 쓴다
    # (처음 실행할 때 모델을 내려받느라 몇 분 걸린다)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # [5단계] 검색기
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # [6~7단계] 프롬프트 및 LLM (Claude)
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}

    Answer in Korean:"""

    prompt = ChatPromptTemplate.from_template(template)
    # Claude 5.5 모델은 temperature 설정을 받지 않는다 (넣으면 400 오류)
    llm = ChatAnthropic(model=ANSWER_MODEL, max_tokens=2000)

    # [8단계] 체인 생성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
