import sys
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()                  # loads OPENAI_API_KEY from .env
client = OpenAI()              # SDK reads the key from the environment

page_num = int(sys.argv[1])
question = sys.argv[2]
text = pymupdf.open("data/RM0090.pdf")[page_num - 1].get_text()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=500,
    messages=[
        {"role": "system",
         "content": "Answer only from the provided context. Cite the page. "
                    "If the answer is not in the context, say you don't know."},
        {"role": "user",
         "content": f"<context page='{page_num}'>\n{text}\n</context>\n\nQuestion: {question}"},
    ],
)
print(response.choices[0].message.content)
print(response.usage)