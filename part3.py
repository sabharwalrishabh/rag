import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini-2025-04-14",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)