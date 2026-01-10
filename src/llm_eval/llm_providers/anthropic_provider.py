import os
import anthropic

from llm_eval.llm_providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, temperature: float = 0.0):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.temperature = temperature

    def generate(self, *, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=512,
        )
        return response.content[0].text
