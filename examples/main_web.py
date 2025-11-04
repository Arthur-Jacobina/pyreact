from pyreact.boot import run_web
from pyreact.core.core import component
from .components import Root
import os
import dotenv
import dspy


@component
def Boot():
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    lm_default = dspy.LM("openai/gpt-4o", api_key=api_key)
    lm_fast = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    models = {
        "default": lm_default,
        "fast": lm_fast,
        "reasoning": lm_default,
    }

    return [Root(key="root", models=models)]


if __name__ == "__main__":
    run_web(Boot, host="127.0.0.1", port=8000)
