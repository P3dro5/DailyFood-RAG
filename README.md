# 🥗 DailyFood RAG — Planeador de Refeições com IA

Aplicação RAG (Retrieval-Augmented Generation) que gera planos de refeições
semanais personalizados com base no tipo de dieta escolhido pelo utilizador.
Utiliza documentos nutricionais reais como base de conhecimento e uma interface
conversacional Gradio para geração e modificação dos planos.

---

## 📋 Funcionalidades

- ✅ Geração de plano semanal completo (Seg → Dom) com 4 refeições por dia
- ✅ Suporte a 5 tipos de dieta: Equilibrada, Mediterrânica, Vegetariana, Vegan, Keto
- ✅ Chat conversacional para modificar o plano (remover ingredientes, substituir refeições, etc.)
- ✅ Mostra um título claro com a dieta gerada no painel do plano
- ✅ Resposta do assistente aparece automaticamente após envio de chat
- ✅ Botão para gerar plano completamente novo
- ✅ Exporta o plano gerado como documento PDF
- ✅ Filtragem de documentos por dieta via metadata no Chroma
- ✅ Histórico de conversa com controlo de tokens (max 5 turns)
- ✅ Logging detalhado de cada etapa do pipeline

---

## 🗂️ Estrutura do Projecto

```
DailyFood_RAG/
│
├── app.py                          # Entry point — inicializa tudo e lança Gradio
│
├── config/
│   └── local.environments          # Variáveis de ambiente (NÃO commitar!)
│
├── core/
│   ├── embeddings.py               # Loader de env + fábricas de modelos
│   ├── vectorstore.py              # Configuração Chroma + helpers de pesquisa
│   ├── ingestion.py                # Pipeline de ingestão de PDFs
│   └── retrieval.py                # RAG chain + pipeline de inferência
│
├── meal_planner/
│   ├── planner.py                  # Orquestrador de planos (gerar / modificar)
│   └── formatter.py                # Formatação Markdown do plano
│
├── ui/
│   └── gradio_interface.py         # Interface Gradio completa
│
├── video/
│   └── video_demonstration.mp4        # Video de demonstração da RAG
│
├── data/
│   └── document_sources.py         # Registo de PDFs e tipos de dieta
│
├── requirements.txt
└── README.md
```

---

## 📚 Fontes de Conhecimento (Base RAG)

| Documento | Organização | Dieta |
|-----------|-------------|-------|
| Dietary Guidelines for Americans 2020-2025 | USDA / HHS | Geral |
| WHO Healthy Diet Guidelines | WHO | Mediterrânica |
| MyPlate Vegetarian Eating Pattern | USDA | Vegetariana |
| Vegan Starter Kit | Physicians Committee (PCRM) | Vegan |
| Ketogenic Diet Review | NIH | Keto |

Todos os documentos são PDFs públicos e de domínio livre carregados via URL.

---

## ⚙️ Instalação e Execução

### 1. Pré-requisitos

- Python >= 3.10
- Conta OpenAI com API Key
- (Opcional) Conta Chroma Cloud para persistência em produção

### 2. Clonar e instalar dependências

```bash
git clone <repo>
cd DailyFood_RAG
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

Edita o ficheiro `config/local.environments` e preenche os valores:

```
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls__...      # Opcional — para tracing LangSmith
CHROMA_API_KEY=...             # Opcional — para Chroma Cloud
CHROMA_TENANT=...              # Opcional — para Chroma Cloud
```

Se `CHROMA_API_KEY` e `CHROMA_TENANT` não estiverem definidos, a app usa
armazenamento local em disco (`./chroma_local_db`) automaticamente.

### 4. Executar

```bash
python app.py
```

A aplicação:
1. Carrega as variáveis de ambiente
2. Inicializa os modelos OpenAI
3. Liga ao vectorstore Chroma
4. Ingere os documentos automaticamente (se a colecção estiver vazia)
5. Abre a interface Gradio em `http://localhost:7860`

---

## 🔄 Fluxo do Pipeline RAG

```
Utilizador escolhe dieta
        ↓
[Gerar Plano]
        ↓
Query enriquecida com contexto de dieta
        ↓
Similarity Search no Chroma (com filtro diet_type)
        ↓
Top-5 chunks relevantes recuperados
        ↓
Prompt: contexto + histórico + query
        ↓
GPT-4o-mini gera plano semanal
        ↓
Formatação Markdown → Gradio UI
        ↓
[Chat] Utilizador pede modificação
        ↓
Plano actual + modificação → novo ciclo RAG
```

### Ingestão de Documentos
O pipeline de ingestão em `core/ingestion.py` descarrega PDFs públicos listados em
`data/document_sources.py`, extrai texto com `PyPDFLoader`, limpa o conteúdo e divide
em chunks. Cada chunk é indexado no vectorstore Chroma com metadata de `diet_type`, o
que permite filtragem de conhecimento relevante por tipo de dieta.

### Inferência RAG
Ao gerar ou modificar um plano, o `core/retrieval.py` pesquisa o Chroma para obter os
chunks mais relevantes para a dieta e o histórico de conversa atual. A chain constrói
um prompt que combina contexto recuperado, intenção do utilizador e histórico de chat.
O LLM gera o plano em texto e `meal_planner/formatter.py` converte esse resultado em
Markdown para exibição na UI e exportação em PDF.

---

## 🏗️ Arquitectura de Componentes

| Componente | Responsabilidade |
|------------|-----------------|
| `core/embeddings.py` | Carrega `.environments`, instancia modelos |
| `core/vectorstore.py` | Chroma Cloud ou local; pesquisa com filtro |
| `core/ingestion.py` | Load PDF → Clean → Chunk → Metadata → Store |
| `core/retrieval.py` | RAG chain, classificação de intenção, inferência |
| `meal_planner/planner.py` | Orquestra gerar/modificar, mantém estado |
| `meal_planner/formatter.py` | Formata output do LLM para Markdown |
| `ui/gradio_interface.py` | UI completa com chat, seletor e painel |
| `data/document_sources.py` | Registo de fontes e mapeamento de dietas |

---

## 🔑 Variáveis de Ambiente

Todas as variáveis estão em `config/local.environments`:

| Variável | Obrigatória | Descrição |
|----------|------------|-----------|
| `OPENAI_API_KEY` | ✅ Sim | Chave API OpenAI |
| `LANGCHAIN_API_KEY` | ❌ Opcional | Tracing LangSmith |
| `LANGCHAIN_PROJECT` | ❌ Opcional | Nome do projecto no LangSmith |
| `LANGCHAIN_TRACING_V2` | ❌ Opcional | Activar tracing (`true`/`false`) |
| `LANGCHAIN_ENDPOINT` | ❌ Opcional | Endpoint LangSmith |
| `CHROMA_API_KEY` | ❌ Opcional | Chave Chroma Cloud |
| `CHROMA_TENANT` | ❌ Opcional | Tenant Chroma Cloud |
| `CHROMA_DATABASE` | ❌ Opcional | Database Chroma Cloud |
| `COLLECTION_NAME` | ❌ Opcional | Nome da colecção (default: `foodDietInformation`) |
| `LLM_MODEL` | ❌ Opcional | Modelo LLM (default: `gpt-4o-mini`) |
| `CLASSIFICATION_MODEL` | ❌ Opcional | Modelo classificação (default: `gpt-3.5-turbo`) |
| `EMBEDDING_MODEL` | ❌ Opcional | Modelo embeddings (default: `text-embedding-3-small`) |

---

## 💬 Exemplos de Uso no Chat

| Pedido | Comportamento |
|--------|--------------|
| "Gera um plano novo" | Gera plano semanal do zero |
| "Remove todas as refeições com frango" | Modifica plano existente |
| "Faz o pequeno-almoço de segunda mais simples" | Modificação cirúrgica |
| "Substitui os jantares por opções mais leves" | Modificação em bloco |
| "Adiciona mais proteína às refeições" | Ajuste nutricional |
| "As refeições de fim de semana podem ser mais elaboradas" | Customização por dia |

---

## 📌 Notas Técnicas

- **Chunking:** 1000 chars com overlap de 200 e separadores hierárquicos
- **Retrieval:** Top-5 chunks com filtro `diet_type` no Chroma; fallback sem filtro
- **Histórico:** Limitado aos últimos 5 turns para controlo de tokens
- **Temperatura LLM:** 0.8 para variedade nos planos; 0.0 na classificação
- **Fallback Chroma:** Se sem credenciais Cloud, usa armazenamento local em disco
