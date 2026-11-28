# 학습 기록

매일 "오늘 한 것 / 막힌 것 / 해결 방법 / 배운 것"을 남긴다. 포트폴리오 문장과 면접 답변의 재료.

## 학습 계획 (약 11일, 하루 3시간)

| 일차 | 주제 | 확인 기준 |
|---|---|---|
| 0-1 | Python 직접 쓰기 (함수, 리스트·딕셔너리, JSON, 파일) | RFP에서 "보안" 줄만 골라 JSON 저장 |
| 0-2 | 타입 힌트, TypedDict, Pydantic, try/except | 요구사항 Pydantic 모델 정의 |
| 1 | LLM API 첫 호출, 토큰·비용 | 호출 비용을 코드로 출력 |
| 2 | 구조화 출력 | RFP → 요구사항 JSON |
| 3 | Tool Calling | 비용 계산 함수를 LLM이 호출 |
| 4~6 | RAG (청킹, 임베딩, Chroma, 출처 표기, Fallback) | RFP Q&A 봇이 출처와 함께 답변 |
| 7~9 | LangGraph (State, Node, Edge, 조건부 분기) | 강의 Module 1~2 |
| 10~11 | 라우터 (1~3번 통합) | 프로토타입 뼈대 완성 |

자료: Claude 한국어 공식 문서(platform.claude.com/docs/ko), 위키독스 「AI 에이전트는 이렇게 만든다 (랭체인부터 랭그래프까지)」, LangChain Academy 「Introduction to LangGraph」

---

## 2026-11-21 (1일차): 첫 LLM 호출, RFP 요구사항 분류

**한 것**
- Claude Console에서 API 키 발급, 크레딧 5달러 충전, 자동 충전 끔
- Colab 보안 비밀에 `ANTHROPIC_API_KEY` 저장 후 첫 호출 성공
- 월드비전 RFP PDF 6~9쪽을 `pypdf`로 추출 → Haiku에게 요구사항 분류 요청 → CSV 저장 (`src/day01_rfp_requirements.py`)

**막힌 것과 해결**

| 에러 | 원인 | 해결 |
|---|---|---|
| `AttributeError: 'ThinkingBlock' object has no attribute 'text'` | 응답 `content`는 블록 리스트이고, Haiku 5.5는 답하기 전에 생각 블록(0번)을 먼저 보냄. `content[0].text`로 무조건 0번을 꺼냄 | `type == "text"`인 블록만 골라 이어 붙임 |
| `EmptyDataError: No columns to parse from file` | `max_tokens=4000`을 생각하는 데 전부 써서 답변(CSV)이 비어 있음. `stop_reason == "max_tokens"` | `max_tokens`를 16000으로 늘림. 상한선일 뿐 쓴 만큼만 과금 |
| CSV에 "이유" 열만 남음 | 답변이 ` ```csv `가 아니라 ` ``` `로 시작 → pandas가 ` ``` `를 헤더로 읽고, 앞 5칸을 인덱스로 돌린 뒤 `index=False`로 저장하며 사라짐 | ` ``` `로 시작하는 줄을 모두 제거 + `df.shape`로 열 개수 검사 |

**배운 것**
- 채팅창에서 분석을 맡기는 것과 코드로 LLM을 부르는 것의 차이: 앱이 대신 하던 단계(텍스트 추출, 요청 구성, 모델 선택, 응답 처리, 비용 기록)를 내가 직접 제어한다. 월드비전 기능은 전부 이 단계 위에 쌓인다.
- LLM 답변 형식은 매번 조금씩 다르다 → 형식 검사 코드가 필요하다. 근본 해결은 구조화 출력(2일차).
- 생각 토큰도 출력 토큰으로 과금된다. 하지만 품질을 위해 생각을 줄이기보다 한도를 넉넉히 준다.
- 같은 모델이면 API든 앱이든 실력은 같다. 결과 차이는 지시문, 맥락, 도구, 설정 차이에서 온다. 같은 조건에서도 매번 결과가 조금씩 달라서 평가셋으로 측정해야 한다.
- 도구는 회의록 기능을 "5주 내 어렵다"며 2단계로 분류했지만, 대표 시나리오라서 1단계로 올림 → MVP 판단은 내 기준으로 고치는 것이 요구사항 분석의 핵심.

## 2026-11-22: 요구사항 분석 정리

**한 것**
- RFP 요구사항 정리 → 원문 대조로 빠진 항목(성능, 관리자, 타 플랫폼 연동, 인프라, 관리자 페이지 접근 제어) 추가
- 목표 세분화 (최종 목표 → G1~G4 → 세부 목표·지표) → `docs/01_rfp_analysis.md`
- 분석 발표 자료 5장 초안을 Figma Slides에서 작성

**배운 것**
- MVP(최소 기능 제품): 핵심 가치를 보여 주는 가장 작은 버전. 빠진 기능은 버리는 게 아니라 "2단계"로 이름을 붙여 로드맵으로 만든다.
- LLMOps: 기록 → 측정 → 문제 발견 → 수정 → 재측정을 반복하는 운영 체계. RFP가 직접 요구함.
- 평가셋 + 코사인 유사도는 지도학습이 아니라 **채점**이다. 모델은 그대로 두고 내 시스템(검색, 프롬프트)을 개선한다. 코사인 유사도는 숫자 오류(3,000원 vs 30,000원)를 못 잡으므로 숫자·키워드 일치 검사나 LLM-as-a-Judge와 섞어 쓴다.

## 2026-11-28: RAG 프로토타입 실행

**한 것**
- LangChain 기본 RAG 예제(PDF 업로드 → FAISS 검색 → LLM 답변, Streamlit 화면)를 로컬에서 실행
- 답변 모델을 Claude로, 임베딩을 로컬 한국어 모델로 바꿔 `src/rag_prototype/`에 정리
- `chunk_size`, `chunk_overlap`, `k`를 `.env`로 분리

**막힌 것과 해결**

| 문제 | 원인 | 해결 |
|---|---|---|
| `Could not open requirements file` | `requirements.txt`가 없는 상위 폴더에서 `pip install` 실행 (압축 해제로 폴더가 두 겹) | `app.py`가 있는 폴더로 이동한 뒤 실행 |
| `(venv)` 표시가 안 붙음 | `python -m venv venv`는 가상환경을 만들기만 함 | `.\venv\Scripts\activate`로 활성화 (PowerShell은 실행 정책 해제 필요) |
| `streamlit` 명령을 찾을 수 없음 | 패키지를 설치하지 않은 다른 가상환경이 켜져 있었음 | 올바른 폴더의 venv를 다시 활성화, `python -m streamlit run app.py`로 실행 |
| `openai.AuthenticationError: 401` | 예제는 OpenAI 전용인데 Claude(Anthropic) 키를 발급받았음. 앱은 예전 OpenAI 키를 읽고 있었음 | 답변 모델을 Claude로 교체, `load_dotenv(override=True)`로 `.env` 값 우선 적용 |

**배운 것**
- OpenAI 키(`sk-proj-`)와 Anthropic 키(`sk-ant-`)는 서로 호환되지 않는다. 라이브러리(`langchain-openai` / `langchain-anthropic`)도 따로 쓴다.
- Claude는 임베딩 API가 없다 → 답변 모델과 임베딩 모델은 따로 고른다. 로컬 임베딩은 API 비용이 없지만 첫 실행이 느리다.
- `load_dotenv()`는 같은 이름의 환경변수가 이미 있으면 `.env`를 무시한다.
- `.env`는 프로그램 시작 때 한 번 읽힌다 → 키를 바꾸면 Streamlit을 재시작해야 한다.
- 가상환경은 한 번 설치하면 남아 있다. 터미널을 새로 열 때마다 활성화만 다시 하면 된다.

