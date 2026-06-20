from typing import Dict

from mlx_lm import generate, load

from markdown_translate_ai.config.models_config import ModelInfo
from markdown_translate_ai.providers.base import APIClient
from markdown_translate_ai.util.statistics import APICallStatistics, TokenUsageTracker


class MlxClient(APIClient):
    """Local MLX API client implementation"""

    def __init__(self, model_info: ModelInfo, stats_tracker: APICallStatistics):
        self.model_info = model_info
        self.stats_tracker = stats_tracker
        self.model, self.tokenizer = load(
            "mlx-community/translategemma-4b-it-4bit_immersive-translate"
        )
        self.token_tracker = TokenUsageTracker()

    def translate(self, prompt: Dict[str, str]) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            response = generate(self.model, self.tokenizer, prompt=prompt, verbose=True)

            # self.stats_tracker.record_call(success=True)
            # self.token_tracker.update(
            #     "openai",
            #     self.model_info.name,
            #     response.usage.prompt_tokens,
            #     response.usage.completion_tokens
            # )

            return response

        except Exception as e:
            self.stats_tracker.record_call(success=False, error=str(e))
            raise

    def cleanup(self) -> None:
        pass
