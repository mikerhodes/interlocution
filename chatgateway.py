from dataclasses import dataclass
from typing import Dict, Generator, List
import os

import ollama
from anthropic import Anthropic


@dataclass
class ModelInfo:
    name: str
    context_length: int


class ChatGateway:
    models: List[str]
    model_to_client: Dict[str, object]

    # For now we hardcode Claude models we want to use
    # while we firm up how ChatGateway should work.
    claude_models: List[str]
    claude_client: Anthropic

    def __init__(self):
        self.models = []
        self.model_to_client = {}

        oc = ollama.Client()
        ollama_models = [model["model"] for model in oc.list()["models"]]
        self.models.extend(ollama_models)
        for m in ollama_models:
            self.model_to_client[m] = oc

        if os.environ.get("ANTHROPIC_API_KEY"):
            self.claude_client = Anthropic()
            self.claude_models = ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"]
            self.models.extend(self.claude_models)

    def list(self) -> List[str]:
        return self.models

    def chat(
        self, model: str, messages: List[Dict[str, str]], stream: bool, num_ctx: int
    ) -> Generator[Dict, None, None]:
        if model in self.claude_models:
            system_prompt = "\n\n".join(
                [m["content"] for m in messages if m["role"] == "system"]
            )
            stream = self.claude_client.messages.create(
                max_tokens=1024,
                system=system_prompt,
                messages=[m for m in messages if not m["role"] == "system"],
                model=model,
                stream=True,
            )
            for event in stream:
                # print(event.type)
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    yield {"message": {"content": event.delta.text}}
        else:
            c = self.model_to_client[model]
            response = c.chat(
                model=model,
                messages=messages,
                stream=True,
                options=ollama.Options(
                    num_ctx=num_ctx,
                ),
            )
            for chunk in response:
                yield chunk

    def show(self, model: str) -> ModelInfo:
        if model in self.claude_models:
            return ModelInfo(model, 200_000)
        else:
            c = self.model_to_client[model]
            m = c.show(model)
            return ModelInfo(model, m.modelinfo[f"{m.details.family}.context_length"])
