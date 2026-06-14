# data/document_sources.py
# =============================================================================
# Registo de fontes documentais para ingestão no pipeline RAG.
# URLs escolhidas por serem leves (< 5MB) e de carregamento rápido.
# =============================================================================

from dataclasses import dataclass
from typing import Literal

DietType = Literal[
    "geral",
    "mediterranica",
    "vegetariana",
    "vegan",
    "keto",
]


@dataclass(frozen=True)
class DocumentSource:
    """Representa uma fonte documental para ingestão."""
    url: str
    diet_type: DietType
    language: str
    description: str


# ---------------------------------------------------------------------------
# Fontes leves (< 5MB) — PDFs públicos e de domínio livre
# ---------------------------------------------------------------------------
DOCUMENT_SOURCES: list[DocumentSource] = [
    DocumentSource(
        url="https://www.fao.org/3/i3325e/i3325e.pdf",
        diet_type="geral",
        language="en",
        description="FAO — Dietary Guidelines and Sustainability",
    ),
    DocumentSource(
        url="https://www.euro.who.int/__data/assets/pdf_file/0020/120166/E73953.pdf",
        diet_type="mediterranica",
        language="en",
        description="WHO Europe — Mediterranean Diet",
    ),
    DocumentSource(
        url="https://www.vegansociety.com/sites/default/files/uploads/downloads/Vegan_Nutrition_for_Athletes.pdf",
        diet_type="vegan",
        language="en",
        description="Vegan Society — Vegan Nutrition Guide",
    ),
    DocumentSource(
        url="https://www.bhf.org.uk/informationsupport/publications/healthy-eating/healthy-eating-booklet",
        diet_type="vegetariana",
        language="en",
        description="BHF — Healthy Eating for Vegetarians",
    ),
    DocumentSource(
        url="https://www.dietdoctor.com/wp-content/uploads/2019/03/Keto-diet-for-beginners.pdf",
        diet_type="keto",
        language="en",
        description="Diet Doctor — Ketogenic Diet Guide",
    ),
]

# Mapeamento conveniente: diet_type -> lista de fontes
SOURCES_BY_DIET: dict[str, list[DocumentSource]] = {}
for _src in DOCUMENT_SOURCES:
    SOURCES_BY_DIET.setdefault(_src.diet_type, []).append(_src)

# Dietas disponíveis para o utilizador seleccionar na UI
AVAILABLE_DIETS: list[str] = [
    "Equilibrada (Geral)",
    "Mediterrânica",
    "Vegetariana",
    "Vegan",
    "Cetogénica (Keto)",
]

# Mapeamento display -> diet_type interno
DIET_DISPLAY_TO_KEY: dict[str, DietType] = {
    "Equilibrada (Geral)": "geral",
    "Mediterrânica": "mediterranica",
    "Vegetariana": "vegetariana",
    "Vegan": "vegan",
    "Cetogénica (Keto)": "keto",
}
