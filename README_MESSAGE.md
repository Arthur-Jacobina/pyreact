# Componentes Log e Message

Este projeto agora suporta dois tipos de componentes para exibição de informações:

## Componente Log

O componente `Log` exibe informações como logs tradicionais, tanto no terminal quanto no browser.

### Uso:
```python
from log import Log

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
from message import Message

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

### No Browser:
- **Log**: Aparece na área de logs (pre com fundo escuro)
- **Message**: Aparece como balões de chat coloridos na área de mensagens

### No Terminal:
- **Log**: Aparece como texto normal
- **Message**: Aparece com cores ANSI baseadas no sender e tipo:
  - User: Azul
  - Assistant: Verde
  - System: Cinza
  - Info: Ciano
  - Warning: Amarelo
  - Error: Vermelho

## Exemplo Completo

Execute o arquivo `example_usage.py` para ver a diferença:

```bash
# No terminal
python example_usage.py

# No browser
python main_web.py
```

## Integração com o Sistema

Os componentes foram integrados nos seguintes locais:

1. **QAHome**: Mensagem de boas-vindas como Message
2. **QAAgent**: Respostas do assistente como Message
3. **GuardRail**: Avisos de toxicidade como Message
4. **Home**: Mensagem de boas-vindas como Message

## Configuração do Socket

O sistema foi modificado para suportar dois tipos de mensagens no socket:

1. **stdout**: Para logs tradicionais
2. **message**: Para mensagens de chat

As mensagens são processadas diferentemente no frontend para renderizar como balões de chat.
