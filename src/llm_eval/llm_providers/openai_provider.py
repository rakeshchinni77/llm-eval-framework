import os
from openai import OpenAI

from llm_eval.llm_providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, temperature: float = 0.0):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature

    def generate(self, *, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content
