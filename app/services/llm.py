from __future__ import annotations

import httpx

from app.core.config import settings
import json

class LLMService:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.llm_model
    
    def health(self) -> bool:
        try:
            response = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=10.0,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:

        payload = {
            "model": self.model,
            "stream": False,
            "keep_alive": "10m",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "format": response_schema,
            "options": {
                "temperature": 0,
            },
        }

        try:

            with httpx.Client(
                timeout=settings.LLM_TIMEOUT
            ) as client:

                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )

                response.raise_for_status()

        except httpx.TimeoutException as exc:

            raise RuntimeError(
                "LLM request timed out."
            ) from exc

        except httpx.HTTPError as exc:

            raise RuntimeError(
                f"LLM request failed: {exc}"
            ) from exc

        try:

            data = response.json()

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from exc

        content = (
            data.get(
                "message",
                {},
            )
            .get(
                "content",
                "",
            )
            .strip()
        )

        if not content:

            raise RuntimeError(
                "LLM returned an empty response."
            )

        try:

            return json.loads(
                content
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "LLM returned invalid structured JSON."
            ) from exc