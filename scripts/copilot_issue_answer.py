import argparse
import openai
import os

parser = argparse.ArgumentParser()
parser.add_argument("--title", required=True)
parser.add_argument("--body", required=True)
parser.add_argument("--output", default="copilot_answer.txt")
args = parser.parse_args()

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = f"""다음은 GitHub Issue에 등록된 질문입니다.

제목: {args.title}
내용: {args.body}

이 질문에 대해 친절하고, 정확한 답변을 한글로 작성해주세요."""

# 최신 openai 라이브러리 방식
client = openai.OpenAI(api_key=openai.api_key)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=350,
    temperature=0.7,
)

answer = response.choices[0].message.content.strip()
with open(args.output, "w", encoding="utf-8") as f:
    f.write(answer)