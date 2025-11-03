from integrations.dspy_integration import DSPyProvider, use_dspy_module
from integrations.use_dspy import use_dspy_call
from log import Log
from message import Message
from pyreact.components.keystroke import Keystroke
from pyreact.core.core import component, hooks
from pyreact.core.provider import create_context
from pyreact.router import (
    Route,
    Router,
    use_route,
    use_navigate,
    use_query_params,
    use_routes_catalog,
)
from pyreact.router.route import use_route_params
from router_agent import RouterAgent as ProjectRouterAgent
import dspy
import os
import dotenv


dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


UserContext = create_context(default="anonymous", name="User")


def use_user():
    user = hooks.use_context(UserContext)

    def _set(u):
        UserContext.set(u)

    return user, _set


class ConvertDates(dspy.Signature):
    """Extract dates from a sentence"""

    sentence: str = dspy.InputField()
    date_from: str = dspy.OutputField()
    date_to: str = dspy.OutputField()


class QASig(dspy.Signature):
    """Answer in one sentence, directly."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


@component
def GuardRail(question, children):
    ver, set_ver = hooks.use_state(0)
    redirected_ver, set_redirected_ver = hooks.use_state(None)
    toxicity = dspy.Predict(
        dspy.Signature(
            "comment -> toxic: bool",
            instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.",
        )
    )
    check_toxicity, result_toxicity, loading, error = use_dspy_call(
        toxicity, model="fast"
    )

    def _check_toxicity():
        if not question.strip():
            return
        check_toxicity(comment=question)

    path, navigate = use_route()

    hooks.use_effect(_check_toxicity, [question])
    hooks.use_effect(
        lambda: set_ver(result_toxicity[1] if result_toxicity else 0), [result_toxicity]
    )

    # Defer navigation to an effect so it happens post-commit (Router mounted)
    def _maybe_redirect():
        if (
            result_toxicity
            and getattr(result_toxicity[0], "toxic", False)
            and redirected_ver != ver
        ):
            set_redirected_ver(ver)
            navigate("/home/3")

    hooks.use_effect(_maybe_redirect, [ver, redirected_ver, result_toxicity])

    # If toxicity check errored, do not block QA – continue and warn
    if error:
        return [
            Message(
                key="toxicity-check-failed",
                text="Não foi possível verificar toxicidade. Continuando.",
                sender="system",
                message_type="warning",
            )
        ] + (children or [])

    if loading:
        return [
            Message(
                key="loading",
                text="Consultando o modelo…",
                sender="system",
                message_type="info",
            )
        ]

    if result_toxicity is None or ver != (result_toxicity or [None, None])[1]:
        return []

    if getattr(result_toxicity[0], "toxic", False):
        return [
            Message(
                key=f"toxic-{result_toxicity[1]}",
                text="Esta pergunta é considerada tóxica",
                sender="system",
                message_type="warning",
            )
        ]
    else:
        return children


@component
def QAAgent(question: str):
    qa_mod = use_dspy_module(QASig, dspy.ChainOfThought, name="qa-cot")
    call_dspy, result, loading, error = use_dspy_call(qa_mod, model="reasoning")

    def _call_dspy():
        if not question.strip():
            return
        call_dspy(question=question)

    hooks.use_effect(_call_dspy, [question])

    if loading:
        return [
            Message(
                key="loading",
                text="Carregando... Aguarde.",
                sender="system",
                message_type="info",
            )
        ]

    if error:
        return [
            Message(
                key="error",
                text=f"Erro: {error}",
                sender="system",
                message_type="error",
            )
        ]

    if result[0] is None:
        return []

    return [
        Message(
            key="agent",
            text=f"{getattr(result[0], 'answer', None)}",
            sender="assistant",
            message_type="chat",
        )
    ]


@component
def QAHome():
    last_message, set_last_message = hooks.use_state("")

    def on_enter(line: str):
        if not line.strip():
            return
        set_last_message(line)

    return [
        Message(
            key="welcome",
            text="Olá! Como posso ajudá-lo hoje?",
            sender="assistant",
            message_type="info",
        ),
        Keystroke(key="qa_input", on_submit=on_enter),
        GuardRail(
            key="guardrail",
            question=last_message,
            children=[QAAgent(key="agent", question=last_message)],
        ),
    ]


@component
def Home():
    route_params = use_route_params()
    query_params = use_query_params()
    navigate = use_navigate()
    user_query, set_user_query = hooks.use_state("")
    catalog = use_routes_catalog()

    hooks.use_effect(
        lambda: print(f"Params: {route_params}, Query: {query_params}"),
        [route_params, query_params],
    )

    def handle_navigate_with_params(k):
        if k == "a":
            navigate(
                "/about", params={"id": "457"}, query={"tab": "profile", "edit": "true"}
            )
        elif k == "q":
            navigate("/qa")
        elif k == "d":
            navigate(
                {
                    "path": "/home/:id",
                    "params": {"id": "789"},
                    "query": {"mode": "debug", "level": "info"},
                    "fragment": "section1",
                }
            )
        else:
            set_user_query(k)

    id_text = (
        f" (ID: {route_params.get('id', 'none')})" if route_params.get("id") else ""
    )
    query_text = f" Query: {query_params}" if query_params else ""

    return [
        Message(
            key="welcome",
            text=f"🏠 Bem-vindo ao Home{id_text}{query_text}",
            sender="system",
            message_type="info",
        ),
        Log(
            key="help",
            text="Press 'a' for about, 'q' for qa, 'd' for dict navigation",
            trigger="mount",
        ),
        Log(
            key="catalog",
            text=f"Rotas disponíveis: {[r.get('name') or r['path'] for r in (catalog or [])]}",
        ),
        Keystroke(key="nav", on_submit=handle_navigate_with_params),
        Message(
            key="instruction",
            text="Digite um comando natural para navegar (ex: 'ir para about' ou 'abrir QA') e pressione Enter:",
            sender="assistant",
            message_type="info",
            trigger="mount",
        ),
        ProjectRouterAgent(
            key="agent-router",
            message=user_query,
            on_navigate=lambda x, y: print(f"Navigating to {x} (ver: {y})"),
        ),
    ]


@component
def About():
    query_params = use_query_params()
    navigate = use_navigate()

    def handle_navigation(k):
        if k == "h":
            navigate("/home/about-redirect", query={"from": "about"})
        elif k == "s":
            navigate("/about", query={"search": "documentation", "filter": "recent"})

    search_text = (
        f" (Search: {query_params.get('search', 'none')})"
        if query_params.get("search")
        else ""
    )
    filter_text = (
        f" Filter: {query_params.get('filter', 'none')}"
        if query_params.get("filter")
        else ""
    )

    return [
        Log(key="a", text=f"ℹ️  About{search_text}{filter_text}", trigger="mount"),
        Log(
            key="help2",
            text="Press 'h' to go home with params, 's' to add search query",
            trigger="mount",
        ),
        Keystroke(key="about-nav", on_submit=handle_navigation),
    ]


@component
def NotFound():
    return [Log(key="404", text="404 – not found", trigger="mount")]


@component
def App():
    return [
        Router(
            initial="/",
            children=[
                Route(
                    key="r1",
                    path="/home/:id",
                    name="home",
                    description="Página inicial com informações gerais",
                    utterances=[
                        "ir para home",
                        "abrir página inicial",
                        "recomeçar",
                        "reiniciar",
                    ],
                    default_params={"id": "1"},
                    children=[Home(key="home")],
                ),
                Route(
                    key="r2",
                    path="/about",
                    name="about",
                    description="Sobre o aplicativo e documentação",
                    utterances=["ir para sobre", "abrir about"],
                    children=[About(key="about")],
                ),
                Route(
                    key="r3",
                    path="/qa",
                    name="qa",
                    description="Perguntas e respostas assistidas sobre questões financeiras",
                    utterances=[
                        "Dúvidas sobre contratos",
                        "Dúvdias sobre pagamento",
                    ],
                    children=[QAHome(key="qa")],
                ),
                Route(
                    key="r4",
                    path="/",
                    name="root",
                    description="Rota raiz (redirect para Home)",
                    utterances=["início", "raiz"],
                    children=[Home(key="home")],
                ),
            ],
        )
    ]


@component
def Root(models=None):
    if models is None:
        lm_default = dspy.LM("openai/gpt-4o", api_key=OPENAI_API_KEY)
        lm_fast = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
        models = {
            "default": lm_default,
            "fast": lm_fast,
            "reasoning": lm_default,
        }
    return [DSPyProvider(key="dspy", models=models, children=[App(key="app")])]
