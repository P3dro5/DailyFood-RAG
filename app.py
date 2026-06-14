# app.py
# =============================================================================
# DailyFood RAG — Entry Point
# =============================================================================
# Ordem de arranque:
#   1. Carregar variáveis de ambiente (config/local.environments)
#   2. Inicializar modelos (embeddings + LLMs)
#   3. Ligar ao vectorstore (Chroma)
#   4. [Opcional] Ingerir documentos se a colecção estiver vazia
#   5. Criar orquestrador de planos
#   6. Lançar interface Gradio
# =============================================================================

import logging
import sys

# ---------------------------------------------------------------------------
# Configuração de logging antes de qualquer import interno
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports internos (após logging configurado)
# ---------------------------------------------------------------------------
from core.embeddings import load_environment, build_embedding_model, build_llm, build_classification_llm
from core.vectorstore import build_vectorstore
from core.ingestion import ingest_all_sources
from meal_planner.planner import MealPlanOrchestrator
from ui.gradio_interface import build_interface


# ---------------------------------------------------------------------------
# Constante — ingerir documentos automaticamente se colecção estiver vazia?
# ---------------------------------------------------------------------------
AUTO_INGEST_IF_EMPTY = True


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap() -> None:
    """
    Inicializa todos os componentes da aplicação e lança o Gradio.

    Raises:
        FileNotFoundError: Se config/local.environments não existir.
        EnvironmentError: Se OPENAI_API_KEY não estiver definida.
        Exception: Qualquer erro crítico durante o arranque.
    """
    logger.info("=" * 70)
    logger.info("  DAILYFOOD RAG — A INICIAR")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # PASSO 1: Carregar variáveis de ambiente
    # ------------------------------------------------------------------
    logger.info("[1/6] Carregando variáveis de ambiente...")
    load_environment("config/local.environments")
    logger.info("  ✓ Ambiente carregado")

    # ------------------------------------------------------------------
    # PASSO 2: Inicializar modelos
    # ------------------------------------------------------------------
    logger.info("[2/6] Inicializando modelos...")
    embeddings = build_embedding_model()
    llm = build_llm(temperature=0.8)                 # Criatividade na geração
    classification_llm = build_classification_llm()  # Determinístico
    logger.info("  ✓ Modelos prontos")

    # ------------------------------------------------------------------
    # PASSO 3: Ligar ao vectorstore
    # ------------------------------------------------------------------
    logger.info("[3/6] Ligando ao vectorstore Chroma...")
    vectorstore = build_vectorstore(embedding_function=embeddings)
    logger.info("  ✓ Vectorstore pronto")

    # ------------------------------------------------------------------
    # PASSO 4: Ingestão de documentos (se necessário)
    # ------------------------------------------------------------------
    logger.info("[4/6] Verificando necessidade de ingestão...")
    if AUTO_INGEST_IF_EMPTY:
        try:
            collection = vectorstore._collection
            count = collection.count()
            logger.info("  Documentos na colecção: %d", count)

            if count == 0:
                logger.info("  Colecção vazia — iniciando ingestão automática...")
                stats = ingest_all_sources(vectorstore)
                logger.info("  ✓ Ingestão concluída: %s", stats)
            else:
                logger.info("  ✓ Colecção já populada — ingestão ignorada")
        except Exception as e:
            logger.warning(
                "  ⚠️  Não foi possível verificar colecção (%s) — "
                "continuando sem ingestão automática", e
            )
    else:
        logger.info("  AUTO_INGEST_IF_EMPTY=False — ingestão ignorada")

    # ------------------------------------------------------------------
    # PASSO 5: Criar orquestrador
    # ------------------------------------------------------------------
    logger.info("[5/6] Criando MealPlanOrchestrator...")
    orchestrator = MealPlanOrchestrator(
        vectorstore=vectorstore,
        llm=llm,
        classification_llm=classification_llm,
    )
    logger.info("  ✓ Orquestrador pronto")

    # ------------------------------------------------------------------
    # PASSO 6: Lançar Gradio
    # ------------------------------------------------------------------
    logger.info("[6/6] Construindo e lançando interface Gradio...")
    demo = build_interface(orchestrator=orchestrator)

    logger.info("=" * 70)
    logger.info("  DAILYFOOD RAG — PRONTO! A abrir no browser...")
    logger.info("=" * 70)

    demo.launch(
        share=True,       # Gera link público temporário (útil para demo)
        debug=True,       # Mostra erros detalhados na UI
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        bootstrap()
    except KeyboardInterrupt:
        logger.info("\n  DailyFood RAG encerrado pelo utilizador.")
    except Exception as exc:
        logger.critical("Erro crítico no arranque: %s", exc, exc_info=True)
        sys.exit(1)
