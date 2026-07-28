"""
Build a knowledge graph from extracted bulletin entities.
Nodes: Bulletin, Region, Country, Situation, Forecast
Saves: data/graph/locust_graph.graphml + graph_summary.json
"""

import json
import re
from pathlib import Path
import networkx as nx

ENTITIES_FILE = Path("G:/FAOLocustKG/data/graph/entities.jsonl")
OUTPUT_DIR = Path("G:/FAOLocustKG/data/graph")
GRAPHML_FILE = OUTPUT_DIR / "locust_graph.graphml"
SUMMARY_FILE = OUTPUT_DIR / "graph_summary.json"

REGION_NAMES = {
    "WESTERN": "Western Region",
    "CENTRAL": "Central Region",
    "EASTERN": "Eastern Region",
}

REGION_COUNTRIES = {
    "WESTERN": [
        "Morocco", "Algeria", "Tunisia", "Libya", "Mauritania",
        "Western Sahara", "Mali", "Niger", "Chad", "Senegal",
    ],
    "CENTRAL": [
        "Egypt", "Sudan", "Eritrea", "Ethiopia", "Djibouti", "Somalia",
        "Saudi Arabia", "Yemen", "Oman", "Kenya", "Uganda",
        "Tanzania", "Mozambique",
    ],
    "EASTERN": [
        "Iran", "Pakistan", "India", "Afghanistan", "Iraq",
        "Syria", "Jordan", "Kuwait", "UAE",
    ],
}


def parse_coverage_year(coverage_period):
    if not coverage_period:
        return None
    m = re.search(r"\b(\d{4})\b", coverage_period)
    return int(m.group(1)) if m else None


def parse_coverage_month(coverage_period):
    if not coverage_period:
        return None
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    for i, month in enumerate(months, 1):
        if month.lower() in coverage_period.lower():
            return month
    return None


def build_graph(records):
    G = nx.DiGraph()

    # Add static region and country nodes
    for region_key, region_name in REGION_NAMES.items():
        G.add_node(region_name, node_type="Region", region_key=region_key)

    for region_key, countries in REGION_COUNTRIES.items():
        region_name = REGION_NAMES[region_key]
        for country in countries:
            if not G.has_node(country):
                G.add_node(country, node_type="Country", region=region_name)
            G.add_edge(region_name, country, edge_type="contains")

    for record in records:
        bulletin_no = record["bulletin_number"]
        if not bulletin_no:
            continue

        coverage = record.get("coverage_period", "")
        year = parse_coverage_year(coverage)
        month = parse_coverage_month(coverage)

        # Bulletin node
        bulletin_id = f"Bulletin_{bulletin_no}"
        G.add_node(
            bulletin_id,
            node_type="Bulletin",
            bulletin_number=bulletin_no,
            issue_date=record.get("issue_date", ""),
            coverage_period=coverage or "",
            forecast_period=record.get("forecast_period", "") or "",
            year=year or 0,
            month=month or "",
            source_file=record.get("source_file", ""),
        )

        # Region situation nodes
        for region_key, alert_level in record.get("regional_alerts", {}).items():
            region_name = REGION_NAMES[region_key]
            situation_id = f"Situation_{bulletin_no}_{region_key}"

            G.add_node(
                situation_id,
                node_type="Situation",
                bulletin_number=bulletin_no,
                region=region_name,
                alert_level=alert_level,
                coverage_period=coverage or "",
            )

            G.add_edge(bulletin_id, situation_id, edge_type="has_situation")
            G.add_edge(situation_id, region_name, edge_type="describes_region")

            # Bulletin covers region
            G.add_edge(bulletin_id, region_name, edge_type="covers")

        # Forecast node (one per bulletin)
        forecast_period = record.get("forecast_period", "") or ""
        if forecast_period:
            forecast_id = f"Forecast_{bulletin_no}"
            G.add_node(
                forecast_id,
                node_type="Forecast",
                bulletin_number=bulletin_no,
                forecast_period=forecast_period,
                coverage_period=coverage or "",
            )
            G.add_edge(bulletin_id, forecast_id, edge_type="has_forecast")

        # Country activity edges
        ha_treated = record.get("ha_treated", {})
        countries_mentioned = record.get("countries_mentioned", [])
        country_detail = record.get("country_detail", {})

        for country in countries_mentioned:
            if not G.has_node(country):
                G.add_node(country, node_type="Country", region="Unknown")

            edge_attrs = {"edge_type": "mentions", "bulletin_number": bulletin_no}

            if country in ha_treated:
                edge_attrs["ha_treated"] = ha_treated[country]

            if country in country_detail:
                detail = country_detail[country]
                edge_attrs["situation_text"] = detail.get("situation", "")[:200]
                edge_attrs["forecast_text"] = detail.get("forecast", "")[:200]

            G.add_edge(bulletin_id, country, **edge_attrs)

    return G


def summarise(G):
    node_types = {}
    for _, data in G.nodes(data=True):
        t = data.get("node_type", "Unknown")
        node_types[t] = node_types.get(t, 0) + 1

    edge_types = {}
    for _, _, data in G.edges(data=True):
        t = data.get("edge_type", "Unknown")
        edge_types[t] = edge_types.get(t, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_types,
    }


def run():
    records = []
    with ENTITIES_FILE.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Building graph from {len(records)} bulletin records...")
    G = build_graph(records)

    summary = summarise(G)
    print(f"\nGraph summary:")
    print(f"  Nodes: {summary['total_nodes']}")
    print(f"  Edges: {summary['total_edges']}")
    print(f"\n  Node types:")
    for t, count in sorted(summary["node_types"].items()):
        print(f"    {t}: {count}")
    print(f"\n  Edge types:")
    for t, count in sorted(summary["edge_types"].items()):
        print(f"    {t}: {count}")

    nx.write_graphml(G, GRAPHML_FILE)
    print(f"\nSaved graph: {GRAPHML_FILE}")

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    run()
