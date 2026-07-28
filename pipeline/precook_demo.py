"""
Pre-generate demo results for all 4 queries (RAG + GraphRAG).
Saves demo_cache.json so the Streamlit app needs no live LLM calls.

Run once (or whenever source data changes):
    python pipeline/precook_demo.py
"""
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import anthropic
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

GRAPHML          = Path("data/graph/locust_graph.graphml")
CHUNKS_FILE      = Path("data/graph/chunks.json")
CHUNK_EMB_FILE   = Path("data/graph/chunk_embeddings.npy")
OUT_FILE         = Path("data/graph/demo_cache.json")

MODEL_NAME  = "all-MiniLM-L6-v2"
LLM_MODEL   = "claude-haiku-4-5"
TOP_K       = 8
MAX_TOKENS  = 600
FAO_URL     = "https://www.fao.org/ag/locusts/en/info/info/index.html"

SYSTEM_PROMPT = (
    "You are an analyst summarising FAO Desert Locust Bulletin data. "
    "Answer in 3-5 concise sentences using only the provided context. "
    "Cite specific bulletin numbers, countries, years, or alert levels. "
    "If context is insufficient, say so clearly."
)

QUERIES = [
    {
        "id": "q1",
        "question": "How did the scale of desert locust control operations change from 2023 to 2024?",
        "graphrag_type": "treatment_yoy",
        "years": [2023, 2024],
    },
    {
        "id": "q2",
        "question": "How did the locust situation differ between the Western and Eastern regions?",
        "graphrag_type": "alert_multi_region",
        "regions": ["WESTERN", "EASTERN"],
    },
    {
        "id": "qW",
        "question": "What was the locust alert situation in the Western Region?",
        "graphrag_type": "alert_multi_region",
        "regions": ["WESTERN"],
    },
    {
        "id": "qC",
        "question": "What was the locust alert situation in the Central Region?",
        "graphrag_type": "alert_multi_region",
        "regions": ["CENTRAL"],
    },
    {
        "id": "qE",
        "question": "What was the locust alert situation in the Eastern Region?",
        "graphrag_type": "alert_multi_region",
        "regions": ["EASTERN"],
    },
    {
        "id": "q3",
        "question": "What ecological and weather conditions were associated with locust upsurges?",
        "graphrag_type": "none",
    },
    {
        "id": "q3b",
        "question": "Where were immature solitarious adults reported during the survey period?",
        "graphrag_type": "none",
    },
    {
        "id": "q4",
        "question": "Which areas received rainfall that could support locust breeding?",
        "graphrag_type": "none",
    },
    {
        "id": "q5",
        "question": "Where were locust hopper bands and swarms reported?",
        "graphrag_type": "none",
    },
    {
        "id": "q6",
        "question": "What vegetation conditions were recorded in the Sahel and Horn of Africa?",
        "graphrag_type": "none",
    },
    {
        "id": "q7",
        "question": "Which countries carried out aerial and ground control operations?",
        "graphrag_type": "none",
    },
    {
        "id": "q8",
        "question": "What were the survey results in the Red Sea coastal areas?",
        "graphrag_type": "none",
    },
    {
        "id": "q9",
        "question": "Where did spring breeding conditions develop in the Western Region?",
        "graphrag_type": "none",
    },
    {
        "id": "qMor",
        "question": "What locust activity and control operations were reported for Morocco?",
        "graphrag_type": "country_mentions",
        "country": "Morocco",
    },
]


# ── chunk RAG helpers ─────────────────────────────────────────────────────────

def chunk_rag(question, chunks, chunk_embs, embed_model, top_k=TOP_K):
    q_emb  = embed_model.encode([question], normalize_embeddings=True)[0]
    scores = chunk_embs @ q_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in top_idx:
        c = chunks[i]
        results.append({
            "chunk_id":        c["chunk_id"],
            "bulletin_number": c["bulletin_number"],
            "page":            c["page"],
            "year":            c["year"],
            "month":           c["month"],
            "coverage_period": c["coverage_period"],
            "score":           round(float(scores[i]), 4),
            "snippet":         c["text"][:400].replace("\n", " "),
        })
    return results


def build_rag_context(chunks_results):
    lines = ["Retrieved bulletin pages (no graph edges followed):"]
    for r in chunks_results:
        lines.append(
            f"  Bulletin {r['bulletin_number']} p{r['page']} "
            f"[{r['coverage_period']}]  score={r['score']:.3f}\n"
            f"  {r['snippet'][:280]}"
        )
    return "\n".join(lines)


# ── GraphRAG traversal helpers ────────────────────────────────────────────────

def traverse_treatment(G, year):
    totals     = defaultdict(lambda: {"ha": 0.0, "bulletins": []})
    trav_edges = []
    trav_nodes = set()

    for node, data in G.nodes(data=True):
        if data.get("node_type") != "Bulletin":
            continue
        if int(data.get("year", 0)) != year:
            continue
        trav_nodes.add(node)

        for _, country, edata in G.out_edges(node, data=True):
            if edata.get("edge_type") != "mentions":
                continue
            ha = float(edata.get("ha_treated", 0) or 0)
            if ha > 0:
                totals[country]["ha"] += ha
                totals[country]["bulletins"].append(int(data.get("bulletin_number", 0)))
                trav_edges.append((node, country))
                trav_nodes.add(country)

    ranked = sorted(totals.items(), key=lambda x: x[1]["ha"], reverse=True)
    return ranked, trav_edges, list(trav_nodes)


def traverse_alerts(G, region_key):
    results    = []
    trav_edges = []
    trav_nodes = set()

    for node, data in G.nodes(data=True):
        if data.get("node_type") != "Situation":
            continue
        if region_key.upper() not in data.get("region", "").upper():
            continue

        for pred in G.predecessors(node):
            pdata = G.nodes[pred]
            if pdata.get("node_type") != "Bulletin":
                continue
            results.append({
                "bulletin":  int(pdata.get("bulletin_number", 0)),
                "year":      int(pdata.get("year", 0)),
                "coverage":  pdata.get("coverage_period", ""),
                "alert":     data.get("alert_level", ""),
            })
            trav_edges.append((pred, node))
            trav_nodes |= {pred, node}

    results.sort(key=lambda x: x["bulletin"])
    return results, trav_edges, list(trav_nodes)


def traverse_country(G, country_name):
    """Traverse all Bulletin nodes that mention a specific country via 'mentions' edges."""
    trav_edges  = []
    trav_nodes  = set()
    data_points = []

    for u, v, edata in G.edges(data=True):
        if edata.get("edge_type") != "mentions":
            continue
        if v.lower() != country_name.lower():
            continue
        bdata = G.nodes[u]
        ha    = float(edata.get("ha_treated", 0) or 0)
        data_points.append({
            "bulletin":   int(bdata.get("bulletin_number", 0)),
            "year":       int(bdata.get("year", 0)),
            "coverage":   bdata.get("coverage_period", ""),
            "ha_treated": ha,
        })
        trav_edges.append((u, v))
        trav_nodes |= {u, v}

    data_points.sort(key=lambda x: x["bulletin"])
    return data_points, trav_edges, list(trav_nodes)


def llm_answer(client, context, question):
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )
    return msg.content[0].text.strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading resources...")
    G          = nx.read_graphml(GRAPHML)
    chunks     = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    chunk_embs = np.load(CHUNK_EMB_FILE)
    embed_model = SentenceTransformer(MODEL_NAME)
    client      = anthropic.Anthropic()

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Chunks: {len(chunks)}, Embeddings: {chunk_embs.shape}")

    cache = {
        "generated_at": date.today().isoformat(),
        "queries": [],
    }

    for q in QUERIES:
        print(f"\n[{q['id']}] {q['question'][:60]}...")

        # ── RAG side ──────────────────────────────────────────────────────────
        print("  RAG: retrieving chunks...")
        rag_chunks  = chunk_rag(q["question"], chunks, chunk_embs, embed_model)
        rag_context = build_rag_context(rag_chunks)
        print("  RAG: calling LLM...")
        rag_summary = llm_answer(client, rag_context, q["question"])

        # ── GraphRAG side ─────────────────────────────────────────────────────
        gtype = q.get("graphrag_type", "none")
        print(f"  GraphRAG: type={gtype}")

        if gtype == "treatment_yoy":
            all_ranked   = {}
            all_edges    = []
            all_nodes    = []
            year_totals  = {}
            for yr in q["years"]:
                ranked, edges, nodes = traverse_treatment(G, yr)
                all_ranked[str(yr)] = [
                    {"country": c, "ha": v["ha"],
                     "bulletins": sorted(set(v["bulletins"]))}
                    for c, v in ranked
                ]
                year_totals[str(yr)] = sum(v["ha"] for _, v in ranked)
                all_edges += edges
                all_nodes += nodes

            lines = []
            for yr in q["years"]:
                lines.append(f"Year {yr} (total {year_totals[str(yr)]:,.0f} ha):")
                for r in all_ranked[str(yr)][:8]:
                    lines.append(f"  {r['country']:<20s} {r['ha']:>6,.0f} ha")
            gr_context = "\n".join(lines)
            gr_summary = llm_answer(client, gr_context, q["question"])
            graphrag = {
                "has_results":    True,
                "traversal_path": f"Bulletin(year) -[mentions]-> Country | years: {q['years']}",
                "nodes_visited":  len(set(all_nodes)),
                "edges_traversed": len(all_edges),
                "type":           "treatment_yoy",
                "year_totals":    year_totals,
                "ranked_by_year": all_ranked,
                "trav_nodes":     list(set(all_nodes)),
                "trav_edges":     [[u, v] for u, v in all_edges],
                "summary":        gr_summary,
            }

        elif gtype == "alert_multi_region":
            all_alerts = {}
            all_edges  = []
            all_nodes  = []
            for region in q.get("regions", []):
                results, edges, nodes = traverse_alerts(G, region)
                all_alerts[region] = results
                all_edges += edges
                all_nodes += nodes

            lines = []
            for region, results in all_alerts.items():
                alert_counts = defaultdict(int)
                for r in results:
                    alert_counts[r["alert"]] += 1
                lines.append(f"{region} region ({len(results)} bulletins):")
                for level, cnt in sorted(alert_counts.items()):
                    lines.append(f"  {level}: {cnt} bulletins")
            gr_context = "\n".join(lines)
            gr_summary = llm_answer(client, gr_context, q["question"])
            graphrag = {
                "has_results":     True,
                "traversal_path":  "Bulletin -[has_situation]-> Situation | regions: "
                                   + ", ".join(q.get("regions", [])),
                "nodes_visited":   len(set(all_nodes)),
                "edges_traversed": len(all_edges),
                "type":            "alert_multi_region",
                "alerts_by_region": all_alerts,
                "trav_nodes":      list(set(all_nodes)),
                "trav_edges":      [[u, v] for u, v in all_edges],
                "summary":         gr_summary,
            }

        elif gtype == "country_mentions":
            country    = q["country"]
            data_pts, edges, nodes = traverse_country(G, country)
            total_ha   = sum(d["ha_treated"] for d in data_pts)
            lines      = [
                f"{country}: {len(data_pts)} bulletin mentions, "
                f"{total_ha:,.0f} ha total treated"
            ]
            for d in data_pts:
                if d["ha_treated"] > 0:
                    lines.append(
                        f"  B{d['bulletin']} ({d['coverage']}): "
                        f"{d['ha_treated']:,.0f} ha"
                    )
            gr_context = "\n".join(lines)
            gr_summary = llm_answer(client, gr_context, q["question"])
            graphrag = {
                "has_results":     len(data_pts) > 0,
                "traversal_path":  f"Bulletin -[mentions]-> {country}",
                "nodes_visited":   len(set(nodes)),
                "edges_traversed": len(edges),
                "type":            "country_mentions",
                "country":         country,
                "total_ha":        total_ha,
                "data_points":     data_pts,
                "trav_nodes":      list(set(nodes)),
                "trav_edges":      [[u, v] for u, v in edges],
                "summary":         gr_summary,
            }

        else:  # "none" — RAG wins, GraphRAG has no traversal path
            graphrag = {
                "has_results":     False,
                "traversal_path":  "No typed traversal path for this query",
                "nodes_visited":   0,
                "edges_traversed": 0,
                "type":            "none",
                "trav_nodes":      [],
                "trav_edges":      [],
                "summary": (
                    "Knowledge graph traversal found no structured path for this query. "
                    "The answer requires synthesising narrative text across bulletin pages "
                    "-- a capability better suited to passage retrieval on the left."
                ),
            }

        cache["queries"].append({
            "id":       q["id"],
            "question": q["question"],
            "rag":      {"summary": rag_summary, "chunks": rag_chunks},
            "graphrag": graphrag,
        })
        print(f"  Done.")

    OUT_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved demo cache -> {OUT_FILE}")


if __name__ == "__main__":
    main()
