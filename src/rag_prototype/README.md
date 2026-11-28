# RAG 프로토타입

PDF를 올리면 문서 내용을 근거로 답하는 RAG 챗봇. 1단계 MVP의 "사내 문서 RAG" 뼈대다.

## 구조

| 파일 | 역할 |
|---|---|
| `rag_module.py` | RAG 파이프라인 8단계 (로드 → 분할 → 임베딩 → FAISS → 검색 → 프롬프트 → Claude → 출력) |
| `app.py` | Streamlit 화면 (PDF 업로드, 채팅) |
| `prompts.py` | 프롬프트 버전 모음 (v0 원래 예제 → v4 형식 지정 + 단계적 사고). `.env`의 `PROMPT_VERSION`으로 선택 |
| `check_retrieval.py` | 검색 진단: 질문별 상위 조각과 정답 조각의 순위 출력 (LLM 호출 없음) |
| `requirements.txt` | 이 폴더 전용 패키지 목록 |

LangChain 기본 RAG 예제 구조에서 시작해 아래를 바꿨다.

- 답변 모델: Claude (`langchain-anthropic`). 모델명은 `.env`의 `ANSWER_MODEL`
- 임베딩: Claude는 임베딩 모델이 없어 로컬 모델 `BAAI/bge-m3` 사용 (API 비용 없음). 처음에 쓴 `ko-sroberta-multitask`보다 검색 순위가 크게 좋아짐 (아래 실험 기록)
- `chunk_size`, `chunk_overlap`, `k`, `search_type`, 프롬프트 버전을 `.env`로 분리해 코드 수정 없이 실험
- 검색된 조각을 "[N쪽] 내용" 형식으로 넘겨 LLM이 출처 쪽수를 쓸 수 있게 함
- 다른 PDF를 올리면 체인을 새로 만들도록 수정, 업로드 파일은 OS 임시 폴더에 저장

## 실행

```bash
pip install -r src/rag_prototype/requirements.txt
cp .env.example .env          # ANTHROPIC_API_KEY 입력
cd src/rag_prototype
streamlit run app.py
```

첫 실행 때 임베딩 모델(약 2GB)을 내려받느라 몇 분 걸린다.

## 실험 기록

질문 "이 사업의 배경과 목적이 뭐야?"에 대해 정답 조각(RFP 2쪽 "사업 배경 및 목적")이 몇 등으로 검색되는지 `check_retrieval.py`로 측정했다.

| 설정 | 정답 조각 순위 | 상위 % | 결론 |
|---|---|---|---|
| chunk 400 / overlap 100 / ko-sroberta | 14 / 75 | 19% | 기준. k=3, 6 모두 "문서에 없다"고 답함 |
| chunk 200 / overlap 50 / ko-sroberta | 26 / 145 | 18% | 청크 크기는 원인이 아님 |
| chunk 400 / overlap 100 / bge-m3 | 1 / 75 | 1% | 임베딩 모델이 핵심 원인 |

bge-m3로 바꾼 뒤 앱에서 같은 질문을 다시 했을 때:

| 질문 | ko-sroberta | bge-m3 |
|---|---|---|
| 이 사업으로 기업이 얻고 싶은 것 3줄 | X "문서에 없다" | O 사업 목적 3줄 요약 |
| 이 사업의 기대효과 | X 제안서 작성 지침으로 답함 | O "기대효과 항목은 없다"고 밝히고 사업 목적을 근거로 답함 (7개 중 5개) |

남은 문제: 목적 항목이 2~3쪽에 걸쳐 있어 3쪽 항목 2개가 빠짐. PDF를 쪽 단위로 읽어 분할하기 때문 → 쪽을 이어 붙인 뒤 분할하는 방식 검토.

관찰: 기존 모델에서는 "사업", "목적"이 반복되는 제안서 작성·입찰 절차 조각(목차 표, 평가 기준, 일정)이 상위 10개를 모두 차지했다.

## 다음 실험

- 프롬프트: 역할 지정, 형식 지정(#명령문·#제약조건·#입력문·#출력형식), 근거 없음 안내, CoT
- 파라미터: `chunk_size` 200/400/800, `chunk_overlap` 0/100/200, `k` 2/3/5, `search_type` similarity·mmr
- 같은 질문 세트로 결과를 비교해 기록
