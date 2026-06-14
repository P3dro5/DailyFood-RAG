# core/ingestion.py
# =============================================================================
# Pipeline de ingestão de documentos PDF.
# Estratégia: download manual com timeout → PyPDFLoader local → fallback texto.
# =============================================================================

import re

import time
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from data.document_sources import DocumentSource, DOCUMENT_SOURCES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
DOWNLOAD_TIMEOUT = 30        # segundos para download do PDF
MAX_PAGES_PER_DOC = 40       # limita páginas para evitar PDFs gigantes


# ---------------------------------------------------------------------------
# Conhecimento embutido — fallback se PDFs falharem
# ---------------------------------------------------------------------------

FALLBACK_KNOWLEDGE: dict[str, str] = {
    "geral": """
Dieta Equilibrada — Princípios Gerais de Nutrição Saudável

Uma dieta equilibrada fornece todos os nutrientes essenciais nas proporções adequadas.
Os principais grupos alimentares são: cereais e leguminosas (base energética),
frutas e vegetais (vitaminas, minerais e fibra), proteínas (carne magra, peixe,
ovos, leguminosas), laticínios (cálcio e proteína) e gorduras saudáveis (azeite,
frutos secos).

Pequeno-almoço equilibrado: inclui proteína (ovos, iogurte), hidratos complexos
(aveia, pão integral) e fruta fresca. Exemplos: papas de aveia com fruta e frutos
secos; torrada integral com ovo estrelado e sumo de laranja natural.

Almoço equilibrado: prato principal com proteína magra (frango grelhado, peixe),
legumes variados e hidratos de carbono complexos (arroz integral, batata-doce,
massa integral). Exemplos: salmão grelhado com arroz integral e brócolos;
peito de frango com batata-doce e salada.

Lanche saudável: iogurte natural com fruta; frutos secos com uma peça de fruta;
hummus com palitos de legumes; batido de fruta com leite vegetal.

Jantar equilibrado: mais leve que o almoço, rico em proteína e vegetais, com
menos hidratos. Exemplos: sopa de legumes + omelete de espinafres; bacalhau
no forno com legumes grelhados; creme de abóbora + salada de atum.

Hidratação: 1,5 a 2 litros de água por dia. Evitar bebidas açucaradas.
Preferir azeite extra-virgem como gordura principal.
Reduzir sal, açúcar refinado e alimentos ultra-processados.
""",

    "mediterranica": """
Dieta Mediterrânica — Padrão Alimentar Tradicional

A dieta mediterrânica é reconhecida pela OMS como um dos padrões alimentares
mais saudáveis do mundo, associada à longevidade e prevenção de doenças
cardiovasculares.

Base da pirâmide: azeite extra-virgem como gordura principal; abundância de
vegetais, frutas, leguminosas, cereais integrais e frutos secos.
Proteína: peixe e marisco 2-3x/semana; aves e ovos moderadamente;
carne vermelha raramente (1-2x/mês).
Laticínios: queijo e iogurte em quantidades moderadas.
Vinho tinto: opcional, com moderação nas refeições.

Pequeno-almoço mediterrânico: pão de fermentação lenta com azeite e tomate;
iogurte grego com mel e nozes; fruta fresca da época.

Almoço mediterrânico: salada grega (tomate, pepino, azeitonas, feta);
peixe grelhado com legumes salteados em azeite; risotto de legumes;
massa com molho de tomate fresco e manjericão.

Lanche mediterrânico: tâmaras com amêndoas; hummus com pão pita integral;
fruta fresca; frutos secos variados.

Jantar mediterrânico: sopa minestrone; bacalhau com grão-de-bico e espinafres;
frango estufado com azeitonas e tomate; ovos mexidos com legumes.

Ervas e especiarias: manjericão, oregãos, alecrim, tomilho, hortelã —
usadas generosamente em substituição do sal.
""",

    "vegetariana": """
Dieta Vegetariana — Alimentação sem Carne

A dieta vegetariana exclui carne e peixe mas pode incluir ovos e laticínios
(lacto-ovo-vegetariana). É nutricionalmente completa quando bem planeada.

Fontes de proteína vegetariana: ovos, laticínios, leguminosas (feijão, grão,
lentilhas, ervilhas), tofu, tempeh, seitan, quinoa, edamame.

Nutrientes a monitorizar: ferro (leguminosas, espinafres, sementes de abóbora),
vitamina B12 (ovos, laticínios ou suplemento), zinco (leguminosas, frutos secos),
cálcio (laticínios, brócolos, amêndoas, bebidas vegetais fortificadas),
ómega-3 (nozes, linhaça, chia).

Pequeno-almoço vegetariano: ovos escalfados com espinafres e torrada integral;
panquecas de aveia com iogurte e fruta; smoothie bowl com granola e sementes;
porridge com canela, maçã e nozes.

Almoço vegetariano: curry de grão-de-bico com arroz basmati; wrap de falafel
com homus e legumes; massa com molho bolonhesa de lentilhas; buddha bowl
com quinoa, tofu e vegetais assados.

Lanche vegetariano: ovos cozidos; iogurte grego com granola; queijo fresco
com fruta; edamame; frutos secos e sementes.

Jantar vegetariano: omelete de legumes; risotto de cogumelos e parmesão;
sopa de lentilhas vermelhas; tacos de feijão preto com guacamole; lasanha
de espinafres e ricota.
""",

    "vegan": """
Dieta Vegan — Alimentação 100% Vegetal

A dieta vegan exclui todos os produtos de origem animal: carne, peixe, ovos,
laticínios e mel. Requer planeamento cuidadoso para garantir todos os nutrientes.

Suplementação obrigatória: vitamina B12 (não existe em plantas).
Suplementação recomendada: vitamina D3 vegan, ómega-3 (óleo de algas),
iodo (sal iodado ou algas), ferro (com vitamina C para absorção).

Fontes de proteína vegan: tofu, tempeh, seitan, edamame, leguminosas (feijão,
lentilhas, grão), quinoa, sementes de cânhamo, proteína de ervilha.

Fontes de cálcio vegan: bebidas vegetais fortificadas, tofu com cálcio, brócolos,
couve-kale, amêndoas, tahini, figos secos.

Pequeno-almoço vegan: papas de aveia com leite de aveia, fruta e sementes de chia;
torrada de pão integral com manteiga de amendoim e banana; smoothie verde com
espinafres, manga e leite de coco; açaí bowl com granola vegan.

Almoço vegan: dahl de lentilhas com arroz integral e naan vegan; burrito de
feijão preto com arroz, guacamole e salsa; pad thai vegan com tofu e amendoins;
salada de quinoa com grão-de-bico assado e tahini.

Lanche vegan: hummus com palitos de cenoura e pepino; bola de energia de
tâmaras e cacau; fruta fresca; iogurte de soja com fruta; frutos secos.

Jantar vegan: sopa de miso com tofu e algas; caril de grão-de-bico e espinafres;
hambúrguer de feijão preto com batata-doce assada; massa com molho de cogumelos
e levedura nutricional (sabor a queijo).
""",

    "keto": """
Dieta Cetogénica (Keto) — Baixo Carboidrato, Alta Gordura

A dieta cetogénica induz cetose metabólica através de restrição severa de
carboidratos (<50g/dia), alta ingestão de gordura (70-75% das calorias) e
proteína moderada (20-25%).

Alimentos permitidos: carnes gordas (bife, bacon, cordeiro), peixe gordo
(salmão, sardinha, atum), ovos, queijos gordos, manteiga, natas, azeite,
óleo de coco, abacate, frutos secos (nozes, amêndoas, macadâmia), vegetais
de baixo carboidrato (espinafres, couve, brócolos, courgette, pepino).

Alimentos a evitar: pão, massa, arroz, batata, fruta (excepto pequenas
quantidades de frutos vermelhos), leguminosas, açúcar, mel, bebidas açucaradas.

Electrólitos: sódio, potássio e magnésio são críticos — suplementar ou consumir
caldo de ossos, abacate e frutos secos regularmente.

Pequeno-almoço keto: ovos estrelados em manteiga com bacon e espinafres;
omelete de queijo com cogumelos e presunto; iogurte grego gordo com frutos
vermelhos e nozes; panquecas de cream cheese e ovos.

Almoço keto: salada caesar com frango grelhado e molho gordo; bife com manteiga
de ervas e salada verde; salmão com espargos salteados em azeite; wrap de alface
com frango, abacate e queijo.

Lanche keto: frutos secos (nozes, amêndoas); ovos cozidos; queijo fatiado;
pepperoni com queijo; abacate com sal e limão; azeitonas.

Jantar keto: costeletas de porco com couve-flor gratinada; frango assado com
brócolos salteados em manteiga; esparguete de courgette (zoodles) com molho
de carne; robalo grelhado com espinafres salteados em alho e azeite.
"""
}


# ---------------------------------------------------------------------------
# Download com timeout
# ---------------------------------------------------------------------------

def _download_pdf_with_timeout(url: str, timeout: int = DOWNLOAD_TIMEOUT) -> Optional[str]:
    """
    Descarrega um PDF para ficheiro temporário com timeout controlado.

    Args:
        url: URL do PDF.
        timeout: Segundos máximos para o download.

    Returns:
        Caminho para o ficheiro temporário, ou None se falhar.
    """
    try:
        logger.info("  Downloading: %s (timeout=%ds)", url, timeout)
        t0 = time.time()

        response = requests.get(
            url,
            timeout=timeout,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (RAG Pipeline)"},
        )
        response.raise_for_status()

        # Guarda em ficheiro temporário
        suffix = ".pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.close()

        elapsed = time.time() - t0
        size_mb = Path(tmp.name).stat().st_size / (1024 * 1024)
        logger.info("  ✓ Download concluído: %.1fMB em %.1fs", size_mb, elapsed)
        return tmp.name

    except requests.exceptions.Timeout:
        logger.warning("  ✗ Timeout após %ds para: %s", timeout, url)
        return None
    except Exception as e:
        logger.warning("  ✗ Erro no download (%s): %s", type(e).__name__, e)
        return None


# ---------------------------------------------------------------------------
# Fallback — texto embutido
# ---------------------------------------------------------------------------

def _load_from_fallback(source: DocumentSource) -> list[Document]:
    """
    Usa o conhecimento nutricional embutido quando o PDF não está disponível.

    Args:
        source: DocumentSource com diet_type.

    Returns:
        Lista com um Document contendo o texto de fallback.
    """
    text = FALLBACK_KNOWLEDGE.get(source.diet_type, "")
    if not text:
        logger.warning("  Sem fallback disponível para diet_type='%s'", source.diet_type)
        return []

    logger.info("  ✓ Usando conhecimento embutido para '%s'", source.diet_type)
    return [Document(
        page_content=text.strip(),
        metadata={"source": f"fallback_{source.diet_type}", "page": 0},
    )]


# ---------------------------------------------------------------------------
# Etapas do pipeline
# ---------------------------------------------------------------------------

def _load_pdf(source: DocumentSource) -> list[Document]:
    """Tenta carregar PDF via download; cai para fallback se falhar."""
    tmp_path = _download_pdf_with_timeout(source.url)

    if tmp_path:
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            # Limita páginas para evitar documentos gigantes
            if len(pages) > MAX_PAGES_PER_DOC:
                logger.info(
                    "  PDF tem %d páginas — limitando a %d",
                    len(pages), MAX_PAGES_PER_DOC
                )
                pages = pages[:MAX_PAGES_PER_DOC]
            logger.info("  ✓ %d páginas carregadas do PDF", len(pages))
            return pages
        except Exception as e:
            logger.warning("  ✗ Erro ao ler PDF (%s) — usando fallback", e)
        finally:
            # Limpa o ficheiro temporário
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return _load_from_fallback(source)


def _clean_documents(documents: list[Document]) -> list[Document]:
    """Limpeza de texto: espaços, numeração de página, strip."""
    for doc in documents:
        text = doc.page_content
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(?<=\.)\s*\d+\s*$", "", text)
        doc.page_content = text.strip()
    return documents


def _chunk_documents(documents: list[Document]) -> list[Document]:
    """Divide em chunks com overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)
    logger.info("  ✓ %d chunks criados", len(chunks))
    return chunks


def _enrich_metadata(chunks: list[Document], source: DocumentSource) -> list[Document]:
    """Adiciona metadata de dieta e fonte a cada chunk."""
    for chunk in chunks:
        chunk.metadata.update({
            "diet_type": source.diet_type,
            "language": source.language,
            "source_description": source.description,
        })
    return chunks


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def ingest_single_source(source: DocumentSource, vectorstore: Chroma) -> tuple[int, str]:
    """
    Pipeline completo para uma fonte: load → clean → chunk → metadata → store.
    Garante fallback para texto embutido se o PDF falhar.
    """
    logger.info("\n%s", "-" * 70)
    logger.info("INGESTÃO: %s", source.description)
    logger.info("Dieta: %s", source.diet_type)

    try:
        pages   = _load_pdf(source)
        pages   = _clean_documents(pages)
        chunks  = _chunk_documents(pages)
        chunks  = _enrich_metadata(chunks, source)

        logger.info("  Armazenando %d chunks no Chroma...", len(chunks))
        vectorstore.add_documents(documents=chunks)
        logger.info("  ✓ Ingestão concluída para '%s'", source.diet_type)

        return len(chunks), source.diet_type

    except Exception as exc:
        logger.error("Erro crítico na ingestão de '%s': %s", source.description, exc, exc_info=True)
        raise


def ingest_all_sources(
    vectorstore: Chroma,
    diet_filter: Optional[list[str]] = None,
) -> dict[str, int]:
    """Ingere todas as fontes registadas, com fallback individual por fonte."""
    stats: dict[str, int] = {}
    sources = DOCUMENT_SOURCES

    if diet_filter:
        sources = [s for s in DOCUMENT_SOURCES if s.diet_type in diet_filter]

    logger.info("\n%s\nINICIANDO INGESTÃO DE %d FONTE(S)\n%s", "="*70, len(sources), "="*70)

    for source in sources:
        try:
            num_chunks, diet_type = ingest_single_source(source, vectorstore)
            stats[diet_type] = stats.get(diet_type, 0) + num_chunks
        except Exception:
            # Tenta fallback directo se o pipeline falhou completamente
            logger.warning("Pipeline falhou — tentando fallback directo para '%s'", source.diet_type)
            try:
                fallback_docs = _load_from_fallback(source)
                if fallback_docs:
                    chunks = _chunk_documents(fallback_docs)
                    chunks = _enrich_metadata(chunks, source)
                    vectorstore.add_documents(documents=chunks)
                    stats[source.diet_type] = stats.get(source.diet_type, 0) + len(chunks)
                    logger.info("  ✓ Fallback bem-sucedido: %d chunks", len(chunks))
            except Exception as fe:
                logger.error("Fallback também falhou para '%s': %s", source.diet_type, fe)

    logger.info("\n%s\nINGESTÃO CONCLUÍDA — %s\n%s", "="*70, stats, "="*70)
    return stats
