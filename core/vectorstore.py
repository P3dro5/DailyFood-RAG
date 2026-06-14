# core/vectorstore.py
# =============================================================================
# Configuração e gestão do Chroma vectorstore.
# Suporta Chroma Cloud (produção) com fallback automático para modo local
# in-memory (desenvolvimento / testes sem credenciais).
# =============================================================================

import os
import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION = "foodDietInformation"


# ---------------------------------------------------------------------------
# Fábrica do vectorstore
# ---------------------------------------------------------------------------

def build_vectorstore(embedding_function: OpenAIEmbeddings) -> Chroma:
    """
    Constrói e devolve o vectorstore Chroma.

    Tenta ligar ao Chroma Cloud se as credenciais estiverem presentes.
    Caso contrário, cai para modo local persistido em disco (ideal para dev).

    Args:
        embedding_function: Modelo de embeddings já instanciado.

    Returns:
        Instância de Chroma configurada.
    """
    collection_name = os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION)
    chroma_api_key = os.getenv("CHROMA_API_KEY", "")
    chroma_tenant = os.getenv("CHROMA_TENANT", "")
    chroma_database = os.getenv("CHROMA_DATABASE", "Daily_Food_RAG")

    if chroma_api_key and chroma_tenant:
        logger.info(
            "Ligando ao Chroma Cloud — tenant='%s', database='%s', collection='%s'",
            chroma_tenant,
            chroma_database,
            collection_name,
        )
        return Chroma(
            embedding_function=embedding_function,
            collection_name=collection_name,
            chroma_cloud_api_key=chroma_api_key,
            tenant=chroma_tenant,
            database=chroma_database,
        )

    # Fallback: persistência local em disco
    persist_dir = "./chroma_local_db"
    logger.warning(
        "Credenciais Chroma Cloud ausentes — usando armazenamento local em '%s'",
        persist_dir,
    )
    return Chroma(
        embedding_function=embedding_function,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


# ---------------------------------------------------------------------------
# Helpers de consulta
# ---------------------------------------------------------------------------

def similarity_search_with_filter(
    vectorstore: Chroma,
    query: str,
    diet_type: Optional[str],
    k: int = 5,
) -> list:
    """
    Executa pesquisa por similaridade com filtro de metadados opcional.

    Se diet_type for fornecido, filtra apenas documentos desse tipo de dieta.
    Caso seja 'geral' ou None, pesquisa sem filtro para cobrir todos os docs.

    Args:
        vectorstore: Instância do Chroma.
        query: Texto da pesquisa.
        diet_type: Tipo de dieta para filtrar (ex: 'keto', 'vegan').
        k: Número de resultados a retornar.

    Returns:
        Lista de Document com os chunks mais relevantes.
    """
    # Sem filtro para pesquisas gerais
    if not diet_type or diet_type == "geral":
        logger.debug("Pesquisa sem filtro de dieta (k=%d)", k)
        return vectorstore.similarity_search(query, k=k)

    chroma_filter = {"diet_type": {"$eq": diet_type}}
    logger.debug("Pesquisa com filtro diet_type='%s' (k=%d)", diet_type, k)

    results = vectorstore.similarity_search(query, k=k, filter=chroma_filter)

    # Fallback: se não houver resultados com filtro, pesquisa sem filtro
    if not results:
        logger.warning(
            "Sem resultados para diet_type='%s' — fallback sem filtro",
            diet_type,
        )
        results = vectorstore.similarity_search(query, k=k)

    return results
