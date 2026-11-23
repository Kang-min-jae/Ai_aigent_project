"""1일차: 첫 LLM API 호출.

목표: 내 코드에서 Claude를 부르고, 답변과 토큰 사용량(비용)을 확인한다.

실행:
    python src/day01_first_call.py
"""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()  # .env 파일의 ANTHROPIC_API_KEY를 환경변수로 불러옴

MODEL = os.getenv("CLASSIFIER_MODEL", "claude-haiku-5-5")

# 100만 토큰당 가격 (달러). 2026년 11월 공개 요금 기준이며, 바뀔 수 있으니 공식 요금표로 확인.
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}


def get_text(message) -> str:
    """응답 블록 중 글(text) 블록만 골라 이어 붙인다.

    Haiku 5.5는 답하기 전에 생각(thinking) 블록을 먼저 보낼 수 있어서
    message.content[0].text 로 꺼내면 AttributeError가 난다.
    """
    return "".join(block.text for block in message.content if block.type == "text")


def main() -> None:
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system="너는 RFP 분석을 돕는 AI 솔루션 아키텍트야.",
        messages=[{"role": "user", "content": "RFP에서 보안 요구사항을 볼 때 무엇을 확인해야 해?"}],
    )

    print(get_text(message))
    print("-" * 40)

    usage = message.usage
    cost = (usage.input_tokens * PRICE_PER_MTOK["input"] + usage.output_tokens * PRICE_PER_MTOK["output"]) / 1_000_000
    print(f"모델: {MODEL}")
    print(f"입력 토큰: {usage.input_tokens} / 출력 토큰: {usage.output_tokens}")
    print(f"예상 비용: ${cost:.5f}")
    print(f"종료 이유: {message.stop_reason}")


if __name__ == "__main__":
    main()
