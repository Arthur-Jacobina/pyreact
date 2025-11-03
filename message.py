from pyreact.core import component, hooks, MessageBuffer
from typing import Literal
import json
import time


@component
def Message(
    text: str,
    trigger: Literal["change", "mount"] = "change",
    sender: Literal["user", "system", "assistant"] = "user",
    message_type: Literal["chat", "info", "warning", "error"] = "chat",
):
    """
    Message component that renders as a chat bubble.

    Args:
        text: Message text
        trigger: When to trigger ("change" or "mount")
        sender: Who sent the message ("user", "system", "assistant")
        message_type: Message type ("chat", "info", "warning", "error")
    """
    deps = [] if trigger == "mount" else [text]

    def _send_message():
        # Sends the message to the console buffer with special formatting
        console = MessageBuffer()

        # Creates a structured message object
        message_data = {
            "type": "message",
            "text": text,
            "sender": sender,
            "message_type": message_type,
            "timestamp": time.time(),
        }

        # Always send as JSON to be processed by the system
        message_json = json.dumps(message_data)
        message_text = f"__MESSAGE__:{message_json}\n"

        # Sends to the console buffer
        console.append(message_text)

    hooks.use_effect(_send_message, deps)
    return []
