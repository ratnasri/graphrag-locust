# FAOLocustKG
A Streamlit dashboard demonstration of knowledge graph Vs GraphRAG side by side
Data used is FAO Desert Locust Bulletin data
## What it shows
Traditional RAG retrieves similar text chunks by embedding similarity. GraphRAG traverses typed edges in a knowledge graph.
This project compares how RAG and multi-hop graph traversal address the same question.

**The key finding:** Some use cases are better addressed using RAG and some are better candidates for GraphRAG.
                     These two techniques complement each other.

## Demo

```
conda activate Agri
streamlit run app.py
```

The dashboard presents four showcase queries, each with a side-by-side RAG answer and GraphRAG answer, a live pyvis knowledge graph, and sidebar filters (year, month, bulletin range, region, country, node type, alert level, edge type).

| Query | Type | RAG | GraphRAG |
|---|---|---|---|
| Morocco locust activity & control | country_mentions traversal | Good | Better (precise ha by bulletin) |
| Rainfall supporting locust breeding | Text retrieval | Good | No structured path |
| Scale of control ops 2023 vs 2024 | treatment_yoy traversal | Partial | Better (yearly totals by country) |
| Ecological conditions and upsurges | Text retrieval | Good | No structured path |

## Dataset
FAO Desert Locust Bulletins No. 532 to 572 (2023 to 2026), 40 issues, ~300 pages.  
https://www.fao.org/ag/locusts/en/info/info/index.html
License: CC BY-NC-SA 3.0 IGO. 

## Graph schema
```
Bulletin
  |-- [has_situation] --> Situation   (alert_level per region: CALM / CAUTION)
  |-- [has_forecast]  --> Forecast
  |-- [covers]        --> Region
  |                           |-- [contains] --> Country
  |-- [mentions]      --> Country     (ha_treated on this edge)
```
224 nodes · 1,639 edges  
Node types: Bulletin (40), Situation (109), Forecast (40), Region (3), Country (32)

## Setup


```
conda create -n Agri python=3.13
conda activate Agri
conda install -c conda-forge pymupdf networkx sentence-transformers anthropic streamlit pyvis
```

Set your Anthropic API key:

```
conda env config vars set ANTHROPIC_API_KEY=sk-ant-...
conda activate Agri
```

## Pipeline
Download PDFs from the FAO url and place them in `data/raw/bulletins`
Run the following pipeline:

```
# Step 1: build the knowledge graph from PDFs
python pipeline/extract_PDFtext.py
python pipeline/split_bulletins.py
python pipeline/extract_entities.py
python pipeline/build_graph.py

# Step 2: build demo cache (chunks + embeddings + LLM answers)
#   Requires internet access to HuggingFace on first run.
#   On subsequent runs with model cached locally:
$env:HF_HUB_OFFLINE = "1"
python pipeline/build_demo.py
```
`build_demo.py` is an orchestrator that runs the pipeline
in sequence and saves all artifacts to `data/graph/`.

Individual pipeline steps:
```
pipeline/chunk_bulletins.py   	 # split bulletin texts into page-level chunks, page 1 skipped  --> chunks.json
pipeline/embed_chunks.py      	 # embed chunks with all-MiniLM-L6-v2 --> chunk_embeddings.npy
pipeline/precook_demo.py      	 # run all queries (RAG + GraphRAG) via LLM -->  demo_cache.json
build_demo.py           		 # orchestrator for chunk -> embed ->precook
```

## Stack
- Python 3.13, conda-forge
- NetworkX (graph), PyMuPDF (PDF extraction), sentence-transformers / all-MiniLM-L6-v2 (embeddings)
- Streamlit + pyvis (dashboard)
- Anthropic Python SDK, claude-haiku-4-5

## License

Pipeline and dashboard code: MIT.  
FAO bulletin data: CC BY-NC-SA 3.0 IGO (not included in this repo).

## Developer notes
- runtime code app.py does not make calls to LLM
- Using a real entity extractor like spaCy gives better results than the regex used in the code
- Graph traversal is limited to pre-defined graphrag types. Data pre-cooking and UI rendering are not dynamic
- graphrag type and tags like regions must be extracted from the chat question, which is hardcoded in the demo
- Demo UI shows only 4 questions whereas pre-cooked data has more. Tweaking VISIBLE_IDs etc in app.py exposes others
