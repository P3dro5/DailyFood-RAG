# core/embeddings.py
# =============================================================================
# Carregamento de variáveis de ambiente a partir de config/local.environments
# e inicialização dos modelos de embedding e LLM.
# =============================================================================

import os
import logging
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carregamento do ficheiro de ambiente
# ---------------------------------------------------------------------------

def load_environment(env_path: str = "config/local.environments") -> None:
    """
    Lê o ficheiro local.environments e injeta as variáveis em os.environ.

    O ficheiro usa o formato KEY=VALUE (uma por linha).
    Linhas começadas por '#' ou vazias são ignoradas.

    Args:
        env_path: Caminho relativo ao root do projecto.
    """
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ficheiro de ambiente não encontrado: {path.resolve()}\n"
            "Cria 'config/local.environments' com base no template fornecido."
        )

    loaded: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            # Ignorar comentários e linhas vazias
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                logger.warning("Linha ignorada (sem '='): %s", line)
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Não sobrescrever variáveis já definidas no sistema
            if key not in os.environ:
                os.environ[key] = value
                loaded.append(key)

    logger.info("Variáveis carregadas de '%s': %s", env_path, loaded)


# ---------------------------------------------------------------------------
# Fábrica de modelos — chamada após load_environment()
# ---------------------------------------------------------------------------

def build_embedding_model() -> OpenAIEmbeddings:
    """
    Instancia o modelo de embeddings com base na variável EMBEDDING_MODEL.

    Returns:
        OpenAIEmbeddings configurado.

    Raises:
        EnvironmentError: Se OPENAI_API_KEY não estiver definida.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY não está definida. "
            "Verifica o ficheiro config/local.environments."
        )

    model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    logger.info("Inicializando embedding model: %s", model_name)

    return OpenAIEmbeddings(model=model_name)


def build_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Instancia o LLM principal para geração de planos de refeição.

    Args:
        temperature: Criatividade da geração (0.0 = determinístico).

    Returns:
        ChatOpenAI configurado.
    """
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    logger.info("Inicializando LLM principal: %s (temp=%.1f)", model_name, temperature)

    return ChatOpenAI(model=model_name, temperature=temperature)


def build_classification_llm() -> ChatOpenAI:
    """
    Instancia o LLM leve usado para classificação de intenção/tópico.
    Temperatura 0 garante respostas determinísticas.

    Returns:
        ChatOpenAI configurado para classificação.
    """
    model_name = os.getenv("CLASSIFICATION_MODEL", "gpt-3.5-turbo")
    logger.info("Inicializando LLM de classificação: %s", model_name)

    return ChatOpenAI(model=model_name, temperature=0)
