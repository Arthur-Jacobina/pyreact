import asyncio
import os
import dotenv
import dspy
from pyreact.boot import bootstrap, read_terminal_and_invoke
from pyreact.core.core import component
from examples.tools import Root


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


async def main():
    myapp = bootstrap(Boot, fps=20)
    try:
        # Messages and errors are rendered from typed events (ConsoleSink);
        # also show LLM and tool activity. :events, :usage, :turns for more.
        await read_terminal_and_invoke(
            myapp,
            prompt="> ",
            wait=True,
            console_kinds=("message", "error", "llm.end", "llm.error", "tool.*"),
        )
    finally:
        myapp.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
