"""1일차: RFP PDF → 요구사항 분류표(CSV).

월드비전 RFP의 요구사항 부분(PDF 6~9쪽)을 추출해 Claude에게 보내고,
요구사항을 분류·MVP 판단한 결과를 CSV로 저장한다. 요구사항 분석 자료의 초안 재료.

실행:
    python src/day01_rfp_requirements.py data/rfp/worldvision_rfp.pdf
    python src/day01_rfp_requirements.py data/rfp/worldvision_rfp.pdf --pages 6-9 --out outputs/requirements.csv

주의: 결과는 초안이다. RFP 원문과 대조해 빠진 항목을 찾고, MVP 판단은 내 기준으로 고친다.
"""

import argparse
import io
import os
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

MODEL = os.getenv("CLASSIFIER_MODEL", "claude-haiku-5-5")
EXPECTED_COLUMNS = ["번호", "분류", "요구사항", "RFP원문근거", "MVP여부", "이유"]

SYSTEM_PROMPT = (
    "너는 RFP를 분석하는 AI 솔루션 아키텍트야. "
    "MVP는 1인 개발자가 5주 안에 만들 수 있는 범위로 판단해. "
    "핵심 기능은 사내 문서 RAG, 라우터 에이전트, 회의록 요약이며 회의록 요약은 대표 시나리오이므로 MVP에 포함해."
)

USER_PROMPT = """아래는 RFP의 요구사항 부분이야. 요구사항을 하나도 빠짐없이 뽑아서 CSV로만 답해 줘.
열: 번호,분류,요구사항,RFP원문근거,MVP여부,이유
- 분류는 기능/관리자/모델/성능/인프라/보안/기타 중 하나
- MVP여부는 "필수" 또는 "2단계"
- 각 칸은 큰따옴표로 감싸고, CSV 말고 다른 설명은 쓰지 마.

{rfp_text}"""


def parse_pages(spec: str) -> range:
    """'6-9' 같은 1부터 세는 쪽 범위를 0부터 세는 range로 바꾼다."""
    start, end = (int(p) for p in spec.split("-"))
    return range(start - 1, end)


def extract_text(pdf_path: Path, pages: range) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(reader.pages[i].extract_text() for i in pages)


def get_text(message) -> str:
    """생각(thinking) 블록은 건너뛰고 글(text) 블록만 이어 붙인다."""
    return "".join(block.text for block in message.content if block.type == "text")


def clean_csv(text: str) -> str:
    """답변을 감싼 코드 블록 표시(```, ```csv 등)를 줄 단위로 모두 제거한다.

    ```csv 만 지우면 ``` 로 시작하는 답변에서 첫 줄이 헤더로 읽혀
    pandas가 앞의 열들을 인덱스로 돌려 버린다 (study-log 2026-11-21 참고).
    """
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="RFP PDF에서 요구사항 분류표를 만든다.")
    parser.add_argument("pdf", type=Path, help="RFP PDF 경로")
    parser.add_argument("--pages", default="6-9", help="요구사항이 있는 쪽 범위 (1부터, 기본 6-9)")
    parser.add_argument("--out", type=Path, default=Path("outputs/requirements.csv"))
    parser.add_argument("--max-tokens", type=int, default=16000, help="생각 토큰까지 포함한 출력 상한")
    args = parser.parse_args()

    rfp_text = extract_text(args.pdf, parse_pages(args.pages))
    print(f"추출한 텍스트: {len(rfp_text):,}자")

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=args.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT.format(rfp_text=rfp_text)}],
    )

    print(f"종료 이유: {message.stop_reason} / 입력 {message.usage.input_tokens} · 출력 {message.usage.output_tokens} 토큰")
    if message.stop_reason == "max_tokens":
        raise SystemExit("토큰 한도에 걸려 답변이 잘렸거나 비어 있어요. --max-tokens를 늘려 다시 실행하세요.")

    csv_text = clean_csv(get_text(message))
    df = pd.read_csv(io.StringIO(csv_text))

    # 형식 검사: LLM 답변 형식은 매번 조금씩 달라서 결과를 그대로 믿지 않는다.
    if list(df.columns) != EXPECTED_COLUMNS:
        raise SystemExit(f"열 구성이 예상과 달라요: {list(df.columns)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")  # 엑셀에서 한글이 깨지지 않게 저장

    print(f"요구사항 {len(df)}개 → {args.out}")
    print(df["분류"].value_counts().to_string())
    print(df["MVP여부"].value_counts().to_string())


if __name__ == "__main__":
    main()
