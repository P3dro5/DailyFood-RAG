# meal_planner/planner.py
# =============================================================================
# Lógica de orquestração do plano de refeições semanal.
# Gere o estado do plano actual e constrói as queries correctas para o RAG.
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from core.retrieval import run_inference
from data.document_sources import DIET_DISPLAY_TO_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modelos de estado
# ---------------------------------------------------------------------------

@dataclass
class MealPlanState:
    """
    Mantém o estado do plano de refeições activo na sessão do utilizador.

    Attributes:
        current_plan: Texto do plano gerado mais recente.
        diet_display_name: Nome da dieta seleccionada para exibição.
        diet_type: Chave interna da dieta (ex: 'keto').
        generation_count: Número de planos gerados na sessão.
    """
    current_plan: str = ""
    diet_display_name: str = "Equilibrada (Geral)"
    diet_type: str = "geral"
    generation_count: int = 0


# ---------------------------------------------------------------------------
# Queries pré-definidas
# ---------------------------------------------------------------------------

def _build_new_plan_query(diet_display_name: str) -> str:
    """Constrói a query para geração de um plano semanal novo."""
    return (
        f"Cria um plano de refeições semanal completo para a dieta {diet_display_name}. "
        f"Inclui todas as refeições de Segunda a Domingo: "
        f"Pequeno-Almoço, Almoço, Lanche e Jantar. "
        f"Cada refeição deve ser variada, saborosa e respeitar os princípios da dieta."
    )


def _build_modification_query(
    modification_request: str,
    current_plan: str,
    diet_display_name: str,
) -> str:
    """Constrói a query para modificação de um plano existente."""
    plan_preview = current_plan[:500] + "..." if len(current_plan) > 500 else current_plan
    return (
        f"Plano actual (para referência):\n{plan_preview}\n\n"
        f"O utilizador quer fazer a seguinte modificação ao plano de dieta {diet_display_name}:\n"
        f"{modification_request}\n\n"
        f"Aplica a modificação mantendo a estrutura semanal completa."
    )


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

class MealPlanOrchestrator:
    """
    Orquestra a geração e modificação de planos de refeições.

    Responsável por:
      - Geração de novos planos semanais
      - Aplicação de modificações a planos existentes
      - Manutenção do estado entre interações
    """

    def __init__(
        self,
        vectorstore: Chroma,
        llm: ChatOpenAI,
        classification_llm: ChatOpenAI,
    ) -> None:
        self.vectorstore = vectorstore
        self.llm = llm
        self.classification_llm = classification_llm
        self._state = MealPlanState()

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def generate_new_plan(
        self,
        diet_display_name: str,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Gera um plano de refeições semanal completamente novo.

        Args:
            diet_display_name: Nome da dieta seleccionada pelo utilizador.
            chat_history: Histórico da conversa Gradio.

        Returns:
            Plano de refeições como string Markdown.
        """
        diet_type = DIET_DISPLAY_TO_KEY.get(diet_display_name, "geral")
        query = _build_new_plan_query(diet_display_name)

        logger.info(
            "Gerando novo plano | dieta=%s | geração #%d",
            diet_type,
            self._state.generation_count + 1,
        )

        response = run_inference(
            query=query,
            diet_type=diet_type,
            diet_display_name=diet_display_name,
            vectorstore=self.vectorstore,
            llm=self.llm,
            classification_llm=self.classification_llm,
            chat_history=chat_history,
        )

        # Actualizar estado
        self._state.current_plan = response
        self._state.diet_display_name = diet_display_name
        self._state.diet_type = diet_type
        self._state.generation_count += 1

        return response

    def modify_plan(
        self,
        modification_request: str,
        diet_display_name: Optional[str] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Modifica o plano de refeições existente com base no pedido do utilizador.

        Se não houver plano actual, gera um novo automaticamente.

        Args:
            modification_request: Descrição da modificação desejada.
            diet_display_name: Dieta (usa a activa se None).
            chat_history: Histórico da conversa Gradio.

        Returns:
            Plano modificado como string Markdown.
        """
        # Se não há plano ativo, gera um novo
        if not self._state.current_plan:
            logger.info("Sem plano ativo, a gerar novo antes de modificar")
            diet = diet_display_name or self._state.diet_display_name
            return self.generate_new_plan(diet, chat_history)

        active_diet = diet_display_name or self._state.diet_display_name
        diet_type = DIET_DISPLAY_TO_KEY.get(active_diet, self._state.diet_type)

        query = _build_modification_query(
            modification_request=modification_request,
            current_plan=self._state.current_plan,
            diet_display_name=active_diet,
        )

        logger.info(
            "A Modificar plano | dieta=%s | pedido='%.60s...'",
            diet_type,
            modification_request,
        )

        response = run_inference(
            query=query,
            diet_type=diet_type,
            diet_display_name=active_diet,
            vectorstore=self.vectorstore,
            llm=self.llm,
            classification_llm=self.classification_llm,
            chat_history=chat_history,
        )

        # Atualizar estado com o plano modificado
        self._state.current_plan = response
        self._state.diet_type = diet_type
        self._state.diet_display_name = active_diet

        return response

    def handle_message(
        self,
        message: str,
        diet_display_name: str,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Ponto de entrada unificado para mensagens do chat.

        Deteta automaticamente se o utilizador quer um plano novo,
        uma modificação ou tem uma pergunta geral.

        Args:
            message: Mensagem do utilizador.
            diet_display_name: Dieta actualmente selecionada.
            chat_history: Histórico da conversa.

        Returns:
            Resposta adequada ao tipo de pedido.
        """
        msg_lower = message.lower()

        # Keywords que indicam pedido de plano completamente novo
        new_plan_keywords = [
            "novo plano", "plano novo", "gerar plano",
            "criar plano", "nova semana", "começar de novo",
            "recomeçar", "plano completo", "semana nova",
        ]

        if any(kw in msg_lower for kw in new_plan_keywords):
            return self.generate_new_plan(diet_display_name, chat_history)

        # Se há plano ativo, trata como modificação
        if self._state.current_plan:
            return self.modify_plan(message, diet_display_name, chat_history)

        # Sem plano ativo, gera um novo
        return self.generate_new_plan(diet_display_name, chat_history)

    @property
    def state(self) -> MealPlanState:
        """Acesso read-only ao estado actual."""
        return self._state
