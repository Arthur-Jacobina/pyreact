# Configuração do DSPy Provider

## Configuração da API Key do OpenAI

Para usar os recursos de IA deste projeto, você precisa configurar sua API key do OpenAI.

### 1. Obter uma API Key

1. Acesse [OpenAI Platform](https://platform.openai.com/)
2. Faça login ou crie uma conta
3. Vá para "API Keys" no menu lateral
4. Clique em "Create new secret key"
5. Copie a chave gerada

### 2. Configurar a API Key

#### Opção A: Variável de Ambiente (Recomendado)

```bash
export OPENAI_API_KEY="sua_api_key_aqui"
```

#### Opção B: Arquivo .env

Crie um arquivo `.env` na raiz do projeto:

```bash
# .env
OPENAI_API_KEY=sua_api_key_aqui
```

**Nota:** O arquivo `.env` está no `.gitignore` por segurança. Nunca commite sua API key no repositório.

### 3. Verificar a Configuração

Para verificar se a configuração está funcionando:

```bash
python -c "import os; print('API Key configurada:', bool(os.getenv('OPENAI_API_KEY')))"
```

### 4. Executar o Projeto

Após configurar a API key, execute o projeto:

```bash
# Para versão web
python main_web.py

# Para versão terminal
python main_terminal.py
```

## Solução de Problemas

### Erro: "DSPy context not available"

Este erro indica que:
1. A API key não está configurada
2. A API key é inválida
3. O DSPyProvider não foi inicializado corretamente

**Soluções:**
1. Verifique se a variável `OPENAI_API_KEY` está definida
2. Confirme que a API key é válida
3. Reinicie o aplicativo após configurar a API key

### Erro: "No language model configured"

Este erro indica que nenhum modelo de linguagem foi configurado no DSPyProvider.

**Solução:** Verifique se o `DSPyProvider` está sendo usado corretamente no componente `Root`.
