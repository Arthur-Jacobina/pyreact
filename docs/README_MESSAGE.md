# Componentes Log e Message

Este projeto agora suporta dois tipos de componentes para exibição de informações:

## Componente Log

O componente `Log` publica o texto como um evento `log` (exemplo de componente de saída próprio).

### Uso:
```python
from examples.log import Log

@component
def MyComponent():
    return [
        Log(key="info", text="Esta é uma informação de log", trigger="mount"),
        Log(key="debug", text="Informação de debug", trigger="change"),
    ]
```

### Parâmetros:
- `text`: Texto a ser exibido
- `trigger`: Quando disparar ("mount" ou "change")
- `key`: Chave única para o componente

## Componente Message

O componente `Message` exibe mensagens como balões de chat no browser e com formatação especial no terminal.

### Uso:
```python
from pyreact.components.message import Message

@component
def MyComponent():
    return [
        Message(
            key="welcome", 
            text="Olá! Bem-vindo!", 
            sender="assistant", 
            message_type="info",
            trigger="mount"
        ),
        Message(
            key="user_msg", 
            text="Esta é uma mensagem do usuário", 
            sender="user", 
            message_type="chat"
        ),
        Message(
            key="warning", 
            text="Atenção!", 
            sender="system", 
            message_type="warning"
        ),
    ]
```

### Parâmetros:
- `text`: Texto da mensagem
- `sender`: Quem enviou ("user", "assistant", "system")
- `message_type`: Tipo da mensagem ("chat", "info", "warning", "error")
- `trigger`: Quando disparar ("mount" ou "change")
- `key`: Chave única para o componente

## Diferenças Visuais

`Message` publica eventos `message`; `Log` (`examples/log.py`) publica eventos
`log` -- quem exibe é o transporte.

### No Browser:
- **Log**: não é encaminhado por padrão (adicione `"log"` a `event_kinds` de `create_fastapi_app`)
- **Message**: Aparece como balões de chat coloridos na área de mensagens

### No Terminal:
- **Log**: Aparece quando `"log"` está em `console_kinds` de `read_terminal_and_invoke`
- **Message**: Aparece com cores ANSI baseadas no sender e tipo:
  - User: Azul
  - Assistant: Verde
  - System: Cinza
  - Info: Ciano
  - Warning: Amarelo
  - Error: Vermelho

## Exemplo Completo

```bash
# No terminal: Message e um evento customizado (Log), sem LM
uv run python -m examples.message
uv run python -m examples.log

# No browser
uv run python -m examples.main_web --app qa_guardrail
```

## Integração com o Sistema

Os componentes foram integrados nos seguintes locais:

1. **QAHome** (`examples/qa_guardrail.py`): Mensagem de boas-vindas como Message
2. **QAAgent** (`examples/qa_guardrail.py`): Respostas do assistente como Message
3. **GuardRail** (`examples/qa_guardrail.py`): Avisos de toxicidade como Message
4. **Home**: Mensagem de boas-vindas como Message

## Configuração do Socket

O sistema foi modificado para suportar dois tipos de mensagens no socket:

1. **stdout**: Para logs tradicionais
2. **message**: Para mensagens de chat

As mensagens são processadas diferentemente no frontend para renderizar como balões de chat.
