# AI Agent Project: 월드비전 AI 업무 지원 플랫폼

실제 공공·비영리 RFP를 바탕으로 **엔터프라이즈 AI 에이전트**를 기획부터 배포까지 혼자 만들어 가는 개인 프로젝트 기록입니다.

- **고객(가상):** 사회복지법인 월드비전, 「2025 월드비전 AI 서비스 플랫폼 구축 사업」 RFP
- **목표:** 직원이 자연어로 사내 정보를 묻고 스스로 해결하는 업무 지원 AI 플랫폼의 MVP를 5주 안에 구축
- **조건:** 1인 개발, 하루 약 3시간(총 약 100시간), API 예산 10달러 이내

## 포트폴리오 스토리

1. **월드비전 RFP**로 범용 엔터프라이즈 에이전트 구조를 5주 동안 설계·구현한다
2. 같은 구조를 **산업안전 도메인**(산학 기업 수요조사 과제를 참고해 자체 수행)으로 이식한다
3. 이식에 걸린 시간과 공통 코드 재사용률을 수치로 남긴다

그래서 처음부터 **도메인에 따라 바뀌는 부분(문서 컬렉션, 프롬프트, 툴 목록, 보고서 템플릿, 마스킹 규칙, 평가셋)을 설정으로 분리**하는 것을 설계 원칙으로 둡니다.

## 진행 현황

| 단계 | 내용 | 결과물 | 상태 |
|---|---|---|---|
| 사전 학습 | LLM API, RAG, LangGraph 기초 | `src/` 연습 코드 | 진행 중 |
| 1. 분석 | RFP 분석, 요구사항 도출, 프로젝트 목표 구체화 | [docs/01_rfp_analysis.md](docs/01_rfp_analysis.md) | 진행 중 |
| 2. 프로토타입 | 아키텍처 설계, 모델 선정, 프로토타입 (RAG + 라우터) | [src/rag_prototype](src/rag_prototype) | 진행 중 |
| 3. 개선·배포 | 평가 기반 개선, 회의록 기능, 배포, 모니터링 | 배포된 서비스 | 예정 |
| 4. 정리 | 결과 발표 자료, 회고 | 발표 자료, 회고 | 예정 |

## MVP 범위 (1단계)

- 사내 문서 RAG (출처 표기, 근거가 없으면 답변 불가 안내)
- 라우터 에이전트 (LangGraph): 문서 검색 / 회의록 요약 / 일반 답변
- 회의록 요약 (STT → 요약 → 참석자별 메일 초안)
- 기본 개인정보 마스킹, 유해 대화 차단
- 배포, 호출별 토큰·비용 기록, 평가셋 기반 개선

2·3단계 범위는 [docs/01_rfp_analysis.md](docs/01_rfp_analysis.md)의 로드맵을 참고하세요.

## 레포 구조

```
.
├── docs/
│   ├── 01_rfp_analysis.md      # RFP 분석, 요구사항, 비용, 목표 세분화
│   ├── 02_topic_selection.md   # 주제 선정 근거와 도메인 확장 계획
│   ├── decision-log.md         # 주요 기술 결정과 이유
│   └── study-log.md            # 날짜별 학습 기록, 막힌 점과 해결 방법
├── src/
│   ├── day01_first_call.py         # 첫 LLM API 호출
│   ├── day01_rfp_requirements.py   # RFP PDF → 요구사항 분류표(CSV)
│   └── rag_prototype/              # PDF 기반 RAG 챗봇 (Streamlit)
├── data/          # RFP 원문 등 로컬 자료 (git에 올리지 않음)
├── outputs/       # 실행 결과물 (git에 올리지 않음)
├── .env.example
└── requirements.txt
```

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # .env에 ANTHROPIC_API_KEY 입력

python src/day01_first_call.py
python src/day01_rfp_requirements.py data/rfp/worldvision_rfp.pdf
```

> RFP 원문 PDF는 저작권과 배포 범위를 고려해 레포에 포함하지 않습니다. `data/rfp/`에 직접 넣어 사용하세요.

## 기술 스택 (계획)

Python · Claude API (Haiku 5.5 / Sonnet 5.5) · LangChain · LangGraph · FAISS·Chroma(벡터 DB) · 로컬 한국어 임베딩 · Streamlit · Docker · 클라우드 배포 · LangSmith 또는 Langfuse(모니터링)
