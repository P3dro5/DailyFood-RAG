# meal_planner/formatter.py
# =============================================================================
# Formatação do plano de refeições para apresentação na UI Gradio.
# Converte o output do LLM em Markdown estruturado e visualmente apelativo.
# =============================================================================

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DAYS_PT = ["Segunda-feira", "Terça-feira", "Quarta-feira",
           "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]

MEAL_ICONS = {
    "pequeno": "🌅",
    "almoço": "🍽️",
    "lanche": "🍎",
    "jantar": "🌙",
}

DIET_EMOJIS = {
    "geral":        "⚖️",
    "mediterranica": "🫒",
    "vegetariana":  "🥦",
    "vegan":        "🌱",
    "keto":         "🥑",
}


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _get_meal_icon(meal_name: str) -> str:
    """Devolve o emoji correspondente ao tipo de refeição."""
    meal_lower = meal_name.lower()
    for key, icon in MEAL_ICONS.items():
        if key in meal_lower:
            return icon
    return "🍴"


def _build_footer(generation_count: int) -> str:
    """Constrói o rodapé com informações de geração."""
    return (
        f"\n\n---\n"
        f"*💡 Dica: Podes pedir modificações específicas no chat — "
        f"por exemplo: \"Remove o frango\" ou \"Adiciona mais proteína ao jantar\".*\n"
        f"*Plano gerado: #{generation_count}*"
    )


# ---------------------------------------------------------------------------
# Formatador principal
# ---------------------------------------------------------------------------

def _build_plan_header(diet_display_name: str, diet_type: str) -> str:
    """Cria o título do plano com emoji e nome da dieta."""
    emoji = DIET_EMOJIS.get(diet_type, "🥗")
    return f"### {emoji} Plano Semanal — {diet_display_name}\n\n"


def format_meal_plan(
    raw_response: str,
    diet_display_name: str,
    diet_type: str,
    generation_count: int = 1,
) -> str:
    """
    Formata a resposta bruta do LLM num Markdown limpo e estruturado.

    O título do plano é renderizado separadamente no painel da UI.

    Args:
        raw_response: Texto devolvido pelo LLM.
        diet_display_name: Nome da dieta para exibição no título.
        diet_type: Chave interna da dieta (para emoji).
        generation_count: Número de gerações na sessão.

    Returns:
        String Markdown formatada e pronta a renderizar no Gradio.
    """
    if not raw_response or not raw_response.strip():
        return "⚠️ Não foi possível gerar o plano. Tenta novamente."

    footer = _build_footer(generation_count)
    formatted = f"{raw_response.strip()}{footer}"

    logger.debug(
        "Plano formatado: %d caracteres | dieta=%s",
        len(formatted), diet_type
    )
    return formatted


def format_plan_title(diet_display_name: str, diet_type: str) -> str:
    """Formata o título do plano para apresentação no painel."""
    return _build_plan_header(diet_display_name, diet_type)


def format_error_message(error: Exception, context: str = "") -> str:
    """
    Formata uma mensagem de erro amigável para o utilizador.

    Args:
        error: Excepção capturada.
        context: Contexto adicional sobre onde ocorreu o erro.

    Returns:
        Mensagem de erro formatada em Markdown.
    """
    logger.error("Erro%s: %s", f" ({context})" if context else "", error, exc_info=True)
    return (
        f"⚠️ **Ocorreu um erro ao gerar o plano.**\n\n"
        f"Detalhe: `{type(error).__name__}: {error}`\n\n"
        f"Por favor, tenta novamente ou verifica as tuas credenciais de API."
    )


def format_loading_message(diet_display_name: str) -> str:
    """Mensagem de carregamento enquanto o plano é gerado."""
    emoji = DIET_EMOJIS.get(diet_display_name, "🥗")
    return (
        f"{emoji} A gerar o teu plano de refeições **{diet_display_name}**...\n\n"
        f"*Isto pode demorar alguns segundos.*"
    )
