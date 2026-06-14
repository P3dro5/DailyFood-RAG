# core/retrieval.py
# =============================================================================
# Pipeline de inferência RAG para geração de planos de refeição.
# Inclui: formatação do histórico, construção da chain e execução do retrieval.
# =============================================================================

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.vectorstore import similarity_search_with_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de prompt
# ---------------------------------------------------------------------------

MEAL_PLAN_SYSTEM_PROMPT = """\
És um nutricionista e chef especializado em planos alimentares semanais saudáveis.

A tua tarefa é criar um plano de refeições semanal completo e detalhado para a dieta especificada.

<instruções>
- Usa EXCLUSIVAMENTE as informações fornecidas no contexto abaixo como base nutricional.
- Organiza o plano de Segunda a Domingo com 4 refeições por dia:
  🌅 Pequeno-Almoço | 🍽️ Almoço | 🍎 Lanche | 🌙 Jantar
- Cada refeição deve incluir prato principal e sugestão de bebida.
- Respeita os princípios da dieta: {diet_display_name}
- Se o utilizador pediu modificações específicas, aplica-as integralmente.
- Usa linguagem clara, apelativa e em Português Europeu.
- Não inventes informações nutricionais que não estejam no contexto.
- Se não houver contexto suficiente, usa conhecimento geral de nutrição mas indica isso.
</instruções>

<contexto_nutricional>
{context}
</contexto_nutricional>

<historico_conversa>
{history}
</historico_conversa>

<pedido_do_utilizador>
{query}
</pedido_do_utilizador>

Plano de Refeições Semanal:
"""

MODIFICATION_INTENT_PROMPT = """\
Classifica a intenção do utilizador numa das seguintes categorias:

- "novo_plano": O utilizador quer um plano completamente novo
- "modificar": O utilizador quer modificar o plano existente  
- "pergunta": O utilizador está a fazer uma pergunta sobre nutrição
- "outro": Outra intenção

Histórico recente:
{history}

Mensagem do utilizador: {query}

Responde APENAS com uma das categorias acima (sem pontuação).
"""


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def format_chat_history(history: list[dict], max_turns: int = 5) -> str:
    """
    Converte o histórico do Gradio (formato messages) em texto legível.

    Limita ao número de turns mais recentes para controlo de tokens.

    Args:
        history: Lista de dicts {"role": str, "content": str}.
        max_turns: Máximo de pares user/assistant a incluir.

    Returns:
        String formatada com o histórico da conversa.
    """
    if not history:
        return "Sem histórico de conversa."

    # Cada turn = 2 mensagens (user + assistant)
    recent = history[-(max_turns * 2):]
    lines = [
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in recent
    ]
    return "\n".join(lines)


def format_context_documents(documents: list) -> str:
    """
    Formata os documentos recuperados como bloco de contexto para o LLM.

    Separa cada chunk com divisor para clareza no prompt.

    Args:
        documents: Lista de Document objects do LangChain.

    Returns:
        String com todo o contexto formatado.
    """
    if not documents:
        return "Sem documentos relevantes encontrados."

    sections = []
    for i, doc in enumerate(documents, 1):
        diet = doc.metadata.get("diet_type", "N/A")
        source = doc.metadata.get("source_description", "N/A")
        sections.append(
            f"[Documento {i} | Dieta: {diet} | Fonte: {source}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

def build_meal_plan_chain(llm: ChatOpenAI):
    """
    Constrói a RAG chain para geração de planos de refeição.

    Args:
        llm: LLM principal instanciado.

    Returns:
        Chain executável (prompt | llm | parser).
    """
    prompt = ChatPromptTemplate.from_template(MEAL_PLAN_SYSTEM_PROMPT)
    return prompt | llm | StrOutputParser()


def build_intent_classification_chain(classification_llm: ChatOpenAI):
    """
    Constrói a chain leve para classificar a intenção do utilizador.

    Args:
        classification_llm: LLM de classificação (temperatura=0).

    Returns:
        Chain executável que devolve a categoria como string.
    """
    prompt = ChatPromptTemplate.from_template(MODIFICATION_INTENT_PROMPT)
    return prompt | classification_llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Pipeline de inferência principal
# ---------------------------------------------------------------------------

def run_inference(
    query: str,
    diet_type: str,
    diet_display_name: str,
    vectorstore: Chroma,
    llm: ChatOpenAI,
    classification_llm: ChatOpenAI,
    chat_history: Optional[list[dict]] = None,
) -> str:
    """
    Executa o pipeline completo de inferência RAG.

    Etapas:
      1. Formatar histórico de conversa
      2. Classificar intenção do utilizador
      3. Construir query enriquecida com contexto de dieta
      4. Recuperar documentos relevantes (com filtro de dieta)
      5. Formatar contexto
      6. Gerar resposta com a RAG chain

    Args:
        query: Mensagem actual do utilizador.
        diet_type: Tipo de dieta interno (ex: 'keto').
        diet_display_name: Nome display da dieta (ex: 'Cetogénica (Keto)').
        vectorstore: Instância Chroma para retrieval.
        llm: LLM principal para geração.
        classification_llm: LLM leve para classificação.
        chat_history: Histórico de conversa no formato Gradio.

    Returns:
        Resposta gerada pelo LLM como string.
    """
    logger.info("=" * 70)
    logger.info("INFERÊNCIA RAG — Dieta: %s | Query: %.60s...", diet_type, query)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # ETAPA 1: Formatar histórico
    # ------------------------------------------------------------------
    formatted_history = format_chat_history(chat_history or [], max_turns=5)
    logger.info("[1/5] Histórico formatado (%d msgs)", len(chat_history or []))

    # ------------------------------------------------------------------
    # ETAPA 2: Classificar intenção
    # ------------------------------------------------------------------
    intent_chain = build_intent_classification_chain(classification_llm)
    intent = intent_chain.invoke({
        "query": query,
        "history": formatted_history,
    }).strip().lower()
    logger.info("[2/5] Intenção detectada: '%s'", intent)

    # ------------------------------------------------------------------
    # ETAPA 3: Construir query enriquecida
    # ------------------------------------------------------------------
    # Enriquece a query com o contexto da dieta para melhor retrieval
    enriched_query = f"dieta {diet_display_name}: {query}"
    logger.info("[3/5] Query enriquecida: '%s'", enriched_query)

    # ------------------------------------------------------------------
    # ETAPA 4: Retrieval com filtro de dieta
    # ------------------------------------------------------------------
    documents = similarity_search_with_filter(
        vectorstore=vectorstore,
        query=enriched_query,
        diet_type=diet_type,
        k=5,
    )
    logger.info("[4/5] %d documentos recuperados", len(documents))

    # ------------------------------------------------------------------
    # ETAPA 5: Formatar contexto e gerar resposta
    # ------------------------------------------------------------------
    context = format_context_documents(documents)
    meal_chain = build_meal_plan_chain(llm)

    response = meal_chain.invoke({
        "context": context,
        "query": query,
        "history": formatted_history,
        "diet_display_name": diet_display_name,
    })

    logger.info("[5/5] Resposta gerada (%d caracteres)", len(response))
    logger.info("=" * 70)

    return response
