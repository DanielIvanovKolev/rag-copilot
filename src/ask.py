"""Ask a question about the STM32 reference manual (level 3: keyword retrieval)."""
import sys

from dotenv import load_dotenv
from openai import OpenAI

from retrieve import load_pages, search

PDF_PATH = "data/RM0090.pdf"
MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "Answer only from the provided context. Cite the page number(s) you used. "
    "If the answer is not in the context, say you don't know."
)


def build_context(hits):
    """Wrap each retrieved page in its own <context> block with its page number."""
    return "\n\n".join(
        f"<context page='{h['page']}'>\n{h['text']}\n</context>" for h in hits
    )


def main():
    question = sys.argv[1]

    pages = load_pages(PDF_PATH)
    hits = search(question, pages, k=5)
    print("Retrieved pages (page, score):", [(h["page"], h["score"]) for h in hits])

    if not hits:
        print("No page contains any of the question's words.")
        return

    load_dotenv()
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{build_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    print()
    print(response.choices[0].message.content)
    print(f"\nTokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")


if __name__ == "__main__":
    main()