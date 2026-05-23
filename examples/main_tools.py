import json
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
    def _on_console(text: str) -> None:
        if not text.startswith("__MESSAGE__:"):
            return
        message_data = json.loads(text[12:])  # strip "__MESSAGE__:"

        sender_colors = {
            "user": "\x1b[34m",  # Blue
            "system": "\x1b[90m",  # Gray
            "assistant": "\x1b[32m",  # Green
        }
        type_colors = {
            "chat": "",
            "info": "\x1b[36m",  # Cyan
            "warning": "\x1b[33m",  # Yellow
            "error": "\x1b[31m",  # Red
        }

        sender = message_data["sender"]
        message_type = message_data["message_type"]
        color = sender_colors.get(sender, "") + type_colors.get(message_type, "")
        reset = "\x1b[0m"
        print(f"{color}[{sender.upper()}] {message_data['text']}{reset}\n", end="", flush=True)

    myapp = bootstrap(Boot, fps=20)
    myapp.attach_web_bridge(
        on_console=_on_console,
        target_loop=asyncio.get_running_loop(),
    )
    await read_terminal_and_invoke(myapp, prompt="> ", wait=True)


if __name__ == "__main__":
    asyncio.run(main())
