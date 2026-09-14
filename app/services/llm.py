from __future__ import annotations

import json

import httpx

from app.core.config import settings


class LLMService:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip('/')
        self.model = settings.llm_model

    def health(self) -> bool:
        try:
            response = httpx.get(
                f'{self.base_url}/api/tags',
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
        response_schema: dict,
        retries: int = 0,
    ) -> dict:
        """Call Ollama and return one validated JSON object.

        Retries are intentionally opt-in so existing callers keep the same
        behaviour. A retry is useful for transient local-server failures or
        an occasionally malformed model response, but it must not change the
        requested schema or silently alter the prompt.
        """
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': user_prompt,
                },
            ],
            'stream': False,
            'format': response_schema,
            'options': {
                'temperature': 0,
            },
        }

        attempts = max(int(retries), 0) + 1
        last_error: Exception | None = None

        for _ in range(attempts):
            try:
                response = httpx.post(
                    f'{self.base_url}/api/chat',
                    json=payload,
                    timeout=settings.llm_timeout_seconds,
                )
                response.raise_for_status()

                data = response.json()
                content = data.get('message', {}).get('content', '')

                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        'Ollama returned an empty response.'
                    )

                parsed = json.loads(content)

                if not isinstance(parsed, dict):
                    raise RuntimeError(
                        'Ollama structured response is not an object.'
                    )

                return parsed

            except (
                httpx.HTTPError,
                json.JSONDecodeError,
                RuntimeError,
                ValueError,
            ) as exc:
                last_error = exc

        if isinstance(last_error, httpx.HTTPError):
            raise RuntimeError(
                f'Ollama request failed: {last_error}'
            ) from last_error

        if last_error is not None:
            raise RuntimeError(
                f'Ollama structured generation failed: {last_error}'
            ) from last_error

        raise RuntimeError('Ollama structured generation failed.')
