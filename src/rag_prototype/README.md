# RAG 프로토타입

PDF를 올리면 문서 내용을 근거로 답하는 RAG 챗봇. 1단계 MVP의 "사내 문서 RAG" 뼈대다.

## 구조

| 파일 | 역할 |
|---|---|
| `rag_module.py` | RAG 파이프라인 8단계 (로드 → 분할 → 임베딩 → FAISS → 검색 → 프롬프트 → Claude → 출력) |
| `app.py` | Streamlit 화면 (PDF 업로드, 채팅) |
| `check_retrieval.py` | 검색 진단: 질문별 상위 조각과 정답 조각의 순위 출력 (LLM 호출 없음) |
| `requirements.txt` | 이 폴더 전용 패키지 목록 |

LangChain 기본 RAG 예제 구조에서 시작해 아래를 바꿨다.

- 답변 모델: Claude (`langchain-anthropic`). 모델명은 `.env`의 `ANSWER_MODEL`
- 임베딩: Claude는 임베딩 모델이 없어 로컬 한국어 모델 `jhgan/ko-sroberta-multitask` 사용 (API 비용 없음)
- `chunk_size`, `chunk_overlap`, `k`를 `.env`로 분리해 코드 수정 없이 실험
- 다른 PDF를 올리면 체인을 새로 만들도록 수정, 업로드 파일은 OS 임시 폴더에 저장

## 실행

```bash
pip install -r src/rag_prototype/requirements.txt
cp .env.example .env          # ANTHROPIC_API_KEY 입력
cd src/rag_prototype
streamlit run app.py
```

첫 실행 때 임베딩 모델(약 400MB)을 내려받느라 몇 분 걸린다.

## 다음 실험

- 프롬프트: 역할 지정, 형식 지정(#명령문·#제약조건·#입력문·#출력형식), 근거 없음 안내, CoT
- 파라미터: `chunk_size` 200/400/800, `chunk_overlap` 0/100/200, `k` 2/3/5, `search_type` similarity·mmr
- 같은 질문 세트로 결과를 비교해 기록
