# ui/gradio_interface.py
# =============================================================================
# Interface Gradio para o DailyFood RAG.
# Inclui: seletor de dieta, chat de conversa, botões de ação e painel de plano.
# =============================================================================

import logging
import tempfile
import unicodedata
from typing import Any, Optional

import gradio as gr
from fpdf import FPDF

from data.document_sources import AVAILABLE_DIETS
from meal_planner.formatter import format_meal_plan, format_error_message, format_plan_title
from meal_planner.planner import MealPlanOrchestrator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSS personalizado
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Container principal */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}

/* Cabeçalho da app */
.app-header {
    text-align: center;
    padding: 20px 0 10px 0;
    border-bottom: 2px solid #e5e7eb;
    margin-bottom: 20px;
}

/* Painel do plano de refeições */
.meal-plan-panel {
    background: #f9fafb;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #e5e7eb;
    min-height: 400px;
    color: #1f2937 !important;
}

.meal-plan-panel * {
    color: #1f2937 !important;
}

/* Botões de ação */
.action-btn {
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.generate-btn {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
}

.clear-btn {
    background: #f3f4f6 !important;
    color: #374151 !important;
}

/* Selector de dieta */
.diet-selector {
    font-weight: 600;
}

/* Chat */
.chatbot-panel {
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
}
"""

# ---------------------------------------------------------------------------
# Textos e exemplos
# ---------------------------------------------------------------------------

APP_TITLE = "🥗 DailyFood — Planeador de Refeições com IA"
APP_DESCRIPTION = """
Gera planos de refeições semanais personalizados com base na tua dieta preferida.
Usa o chat para modificar o plano, remover ingredientes ou pedir alternativas.
"""

EXAMPLE_MESSAGES = [
    "Gera um plano novo para esta semana",
    "Remove todas as refeições com carne vermelha",
    "Substitui os almoços de quarta e quinta por opções vegetarianas",
    "Adiciona mais proteína ao pequeno-almoço",
    "Preciso de opções sem glúten para o jantar",
    "Faz as refeições de fim de semana mais elaboradas",
]


# ---------------------------------------------------------------------------
# Builder da interface
# ---------------------------------------------------------------------------

def build_interface(orchestrator: MealPlanOrchestrator) -> gr.Blocks:
    """
    Constrói e devolve a interface Gradio completa.

    Arquitectura da UI:
      - Coluna Esquerda: Seletor de dieta + Botões + Exemplos
      - Coluna Direita: Painel do plano gerado
      - Baixo: Chat para modificações

    Args:
        orchestrator: Instância do MealPlanOrchestrator já configurada.

    Returns:
        Instância gr.Blocks pronta para .launch().
    """

    # ------------------------------------------------------------------
    # Helpers e handlers de eventos
    # ------------------------------------------------------------------

    def _normalize_plan_text(plan_markdown: str) -> str:
        # Remove markdown emphasis and headings
        cleaned = (
            plan_markdown
            .replace("**", "")
            .replace("### ", "")
            .replace("## ", "")
            .replace("*", "")
        )
        # Replace common Unicode punctuation that FPDF (latin-1) can't encode
        replacements = {
            "\u2014": "-",  # em dash
            "\u2013": "-",  # en dash
            "\u2018": "'",  # left single quote
            "\u2019": "'",  # right single quote
            "\u201c": '"',  # left double quote
            "\u201d": '"',  # right double quote
            "\u2026": "...",  # ellipsis
            "\u2022": "-",  # bullet
            "\u00A0": " ",  # non-breaking space
        }
        for k, v in replacements.items():
            cleaned = cleaned.replace(k, v)
        # Normalize accents and remove any remaining characters unsupported by latin-1
        normalized = unicodedata.normalize("NFKD", cleaned)
        safe = normalized.encode("latin-1", errors="ignore").decode("latin-1")
        return safe.strip()

    def _create_plan_pdf(plan_markdown: str, diet_display_name: str) -> str:
        normalized = _normalize_plan_text(plan_markdown)
        safe_title = _normalize_plan_text(f"Plano Semanal - {diet_display_name}")
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, safe_title, ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=11)

        for line in normalized.splitlines():
            pdf.multi_cell(0, 8, line)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.close()
        pdf.output(temp_file.name)
        return temp_file.name

    def on_generate_plan(
        diet_display_name: str,
        history: list[dict],
    ) -> tuple[str, str, list[dict], str, str, Any]:
        """
        Handler do botão "Gerar Novo Plano".
        Devolve: (título, plano_markdown, historico_atualizado, status)
        """
        try:
            logger.info("UI: Gerar novo plano | dieta=%s", diet_display_name)
            plan = orchestrator.generate_new_plan(
                diet_display_name=diet_display_name,
                chat_history=history,
            )
            title = format_plan_title(
                diet_display_name=diet_display_name,
                diet_type=orchestrator.state.diet_type,
            )
            formatted = format_meal_plan(
                raw_response=plan,
                diet_display_name=diet_display_name,
                diet_type=orchestrator.state.diet_type,
                generation_count=orchestrator.state.generation_count,
            )
            # Adiciona ao histórico do chat
            new_history = history + [
                {"role": "user", "content": f"Gera um plano semanal para dieta {diet_display_name}"},
                {"role": "assistant", "content": formatted},
            ]
            return (
                title,
                formatted,
                new_history,
                f"✅ Plano gerado com sucesso! (#{orchestrator.state.generation_count})",
                formatted,
                gr.update(interactive=True),
            )
        except Exception as e:
            err_msg = format_error_message(e, "geração de plano")
            return "", err_msg, history, "❌ Erro ao gerar plano", "", gr.update(interactive=False)

    def on_chat_user_message(
        message: str,
        history: list[dict],
    ) -> tuple[list[dict], str]:
        """
        Atualiza o chat imediatamente com a mensagem do utilizador.
        """
        if not message.strip():
            return history, "⚠️ Mensagem vazia"

        new_history = history + [{"role": "user", "content": message}]
        return new_history, "⌛ A processar a tua mensagem..."

    def on_chat_response(
        message: str,
        history: list[dict],
        diet_display_name: str,
    ) -> tuple[str, str, list[dict], str, str, Any]:
        """
        Processa a mensagem de chat e adiciona a resposta ao histórico.
        """
        if not message.strip():
            return "", gr.update(), history, "⚠️ Mensagem vazia", "", gr.update(interactive=False)

        try:
            logger.info("UI: Chat message | dieta=%s | msg='%.60s'", diet_display_name, message)

            response = orchestrator.handle_message(
                message=message,
                diet_display_name=diet_display_name,
                chat_history=history,
            )
            title = format_plan_title(
                diet_display_name=diet_display_name,
                diet_type=orchestrator.state.diet_type,
            )
            formatted = format_meal_plan(
                raw_response=response,
                diet_display_name=diet_display_name,
                diet_type=orchestrator.state.diet_type,
                generation_count=orchestrator.state.generation_count,
            )
            new_history = history + [
                {"role": "assistant", "content": formatted},
            ]
            return (
                title,
                formatted,
                new_history,
                "✅ Plano atualizado",
                formatted,
                gr.update(interactive=True),
            )
        except Exception as e:
            err_msg = format_error_message(e, "chat")
            new_history = history + [
                {"role": "assistant", "content": err_msg},
            ]
            return "", gr.update(), new_history, "❌ Erro ao processar mensagem", "", gr.update(interactive=False)

    def on_clear_chat() -> tuple[list, str, str, str, str, Any]:
        """Handler do botão Limpar. Reseta chat, painel e título."""
        return [], "", "", "🔄 Conversa limpa", "", gr.update(interactive=False)

    def on_export_pdf(
        plan_text: str,
        diet_display_name: str,
    ) -> tuple[Optional[str], str]:
        if not plan_text or not plan_text.strip():
            return None, "❌ Nenhum plano disponível para exportar."

        try:
            pdf_path = _create_plan_pdf(plan_text, diet_display_name)
            return pdf_path, "✅ PDF pronto para download!"
        except Exception as e:
            logger.exception("Erro ao criar PDF do plano", exc_info=e)
            err_msg = format_error_message(e, "exportação para PDF")
            return None, err_msg

    def on_example_click(example: str) -> str:
        """Preenche o textbox com o exemplo clicado."""
        return example

    # ------------------------------------------------------------------
    # Layout da interface
    # ------------------------------------------------------------------

    with gr.Blocks(css=CUSTOM_CSS, title="DailyFood RAG") as demo:

        # Cabeçalho
        gr.HTML(f"""
            <div class="app-header">
                <h1>🥗 DailyFood</h1>
                <p style="color: #6b7280; font-size: 1.1em;">
                    Planeador de Refeições Semanal com IA — RAG Powered
                </p>
            </div>
        """)

        # Estado partilhado entre componentes
        plan_state = gr.State("")

        with gr.Row():
            # ----------------------------------------------------------
            # COLUNA ESQUERDA — Controlos
            # ----------------------------------------------------------
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### ⚙️ Configurações")

                diet_selector = gr.Dropdown(
                    choices=AVAILABLE_DIETS,
                    value=AVAILABLE_DIETS[0],
                    label="🥗 Tipo de Dieta",
                    info="Seleciona a dieta para o teu plano semanal",
                    elem_classes=["diet-selector"],
                )

                generate_btn = gr.Button(
                    "🚀 Gerar Novo Plano Semanal",
                    variant="primary",
                    elem_classes=["action-btn", "generate-btn"],
                    size="lg",
                )

                clear_btn = gr.Button(
                    "🗑️ Limpar Conversa",
                    variant="secondary",
                    elem_classes=["action-btn", "clear-btn"],
                )

                status_box = gr.Textbox(
                    label="Estado",
                    value="Pronto. Seleciona uma dieta e gera o teu plano!",
                    interactive=False,
                    max_lines=2,
                )

                export_btn = gr.Button(
                    "📄 Exportar como PDF",
                    variant="secondary",
                    elem_classes=["action-btn"],
                    size="lg",
                    interactive=False,
                )

                pdf_file = gr.File(
                    label="Download do PDF",
                    file_count="single",
                    interactive=False,
                )

                gr.Markdown("---")
                gr.Markdown("### 💬 Exemplos de Pedidos")

                for example in EXAMPLE_MESSAGES:
                    ex_btn = gr.Button(
                        f"→ {example}",
                        variant="secondary",
                        size="sm",
                    )
                    # Guardar referência para event binding abaixo
                    ex_btn._example_text = example

            # ----------------------------------------------------------
            # COLUNA DIREITA — Plano de Refeições
            # ----------------------------------------------------------
            with gr.Column(scale=2):
                gr.Markdown("### 📋 Plano de Refeições")

                plan_title = gr.Markdown(
                    value="",
                    elem_classes=["meal-plan-panel-title"],
                )

                plan_output = gr.Markdown(
                    value=(
                        "*O teu plano semanal aparecerá aqui após geração.*\n\n"
                        "👈 Seleciona uma dieta e clica em **Gerar Novo Plano Semanal**."
                    ),
                    elem_classes=["meal-plan-panel"],
                )

        # ----------------------------------------------------------
        # SECÇÃO DE CHAT — Modificações
        # ----------------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("### 💬 Chat — Modificações e Perguntas")
        gr.Markdown(
            "*Usa o chat para modificar o plano, remover ingredientes, "
            "adicionar alternativas ou fazer perguntas sobre nutrição.*"
        )

        chatbot = gr.Chatbot(
            value=[],
            label="Conversa com o DailyFood Assistant",
            height=350,
            elem_classes=["chatbot-panel"],
        )

        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ex: Remove o frango e substitui por tofu...",
                label="A tua mensagem",
                scale=5,
                max_lines=3,
            )
            send_btn = gr.Button("📨 Enviar", variant="primary", scale=1)

        # ----------------------------------------------------------
        # Bindings de eventos
        # ----------------------------------------------------------

        # Gerar novo plano
        generate_btn.click(
            fn=on_generate_plan,
            inputs=[diet_selector, chatbot],
            outputs=[plan_title, plan_output, chatbot, status_box, plan_state, export_btn],
        )

        # Enviar mensagem (botão)
        send_btn.click(
            fn=on_chat_user_message,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, status_box],
        ).then(
            fn=on_chat_response,
            inputs=[chat_input, chatbot, diet_selector],
            outputs=[plan_title, plan_output, chatbot, status_box, plan_state, export_btn],
        ).then(
            fn=lambda: "",
            outputs=[chat_input],
        )

        # Enviar mensagem (Enter)
        chat_input.submit(
            fn=on_chat_user_message,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, status_box],
        ).then(
            fn=on_chat_response,
            inputs=[chat_input, chatbot, diet_selector],
            outputs=[plan_title, plan_output, chatbot, status_box, plan_state, export_btn],
        ).then(
            fn=lambda: "",
            outputs=[chat_input],
        )

        # Limpar conversa
        clear_btn.click(
            fn=on_clear_chat,
            outputs=[chatbot, plan_title, plan_output, status_box, plan_state, export_btn],
        )

        # Exportar como PDF
        export_btn.click(
            fn=on_export_pdf,
            inputs=[plan_state, diet_selector],
            outputs=[pdf_file, status_box],
        )

        # Exemplos — binding dinâmico
        for child in demo.blocks.values():
            if hasattr(child, "_example_text"):
                child.click(
                    fn=lambda txt=child._example_text: txt,
                    outputs=[chat_input],
                )

    return demo
