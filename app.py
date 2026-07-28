"""
Desert Locust Bulletin Dashboard
Streamlit app: RAG (chunk retrieval) vs Knowledge Graph Traversal
over FAO Desert Locust Bulletins No. 532-572, 2023-2026
"""
import json
from pathlib import Path

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# ── paths ─────────────────────────────────────────────────────────────────────
GRAPHML      = Path("data/graph/locust_graph.graphml")
CACHE_FILE   = Path("data/graph/demo_cache.json")
FAO_URL      = "https://www.fao.org/ag/locusts/en/info/info/index.html"

# ── colours ───────────────────────────────────────────────────────────────────
SITUATION_COLOR = {
    "CALM":    "#3CB034",  # parrot green
    "CAUTION": "#E67E22",  # amber-orange
    "WARNING": "#FF5722",  # deep orange-red
    "ALERT":   "#FF1744",  # vivid red
    "CRISIS":  "#FF006E",  # hot magenta-red — bright, not dark
}
TYPE_COLOR = {
    "Bulletin":  "#808080",
    "Region":    "#6B2FA0",
    "Country":   "#C9A8E0",
    "Forecast":  "#00BCD4",
    "Situation": "#3CB034",  # overridden per node by alert level
}
GOLDENROD  = "#DAA520"
EDGE_COLOR = {
    "mentions":         "#CCCCCC",
    "has_situation":    "#AAAAAA",
    "has_forecast":     "#AAAAAA",
    "covers":           "#AAAAAA",
    "describes_region": "#AAAAAA",
    "contains":         "#AAAAAA",
}
EDGE_ACTV = "#DAA520"

def clean_summary(text):
    """Strip markdown heading lines that LLMs tend to generate."""
    if not text:
        return ""
    lines = [l for l in text.splitlines() if not l.strip().startswith("#")]
    return "\n".join(lines).strip()

ALL_ALERT_LEVELS = ["CALM", "CAUTION", "WARNING", "ALERT", "CRISIS"]
ALL_EDGE_TYPES   = ["mentions", "has_situation", "has_forecast",
                    "covers", "describes_region", "contains"]
MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]

FILTER_DEFAULTS = {
    "year_sel":     "All",
    "month_sel":    [],
    "bul_min":      532,
    "bul_max":      572,
    "region_sel":   "All",
    "country_sel":  [],
    "top_k_sl":     4,
    "min_score_sl": 0.65,
    **{f"nt_{t}": True for t in ["Bulletin","Region","Country","Forecast"]},
    "path_only": False,
    **{f"al_{l}": True for l in ["CALM","CAUTION","WARNING","ALERT","CRISIS"]},
    **{f"et_{e}": True for e in ["mentions","has_situation","has_forecast",
                                  "covers","describes_region","contains"]},
}

KNOWN_COUNTRIES = [
    "Morocco","Algeria","Tunisia","Libya","Mauritania","Western Sahara",
    "Mali","Niger","Chad","Senegal","Egypt","Sudan","Eritrea","Ethiopia",
    "Djibouti","Somalia","Saudi Arabia","Yemen","Oman","Iran","Pakistan",
    "India","Kenya","Uganda","Tanzania","Mozambique","Afghanistan",
    "Iraq","Syria","Jordan","Kuwait","UAE",
]


# ── resource loading ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading graph...")
def load_graph():
    return nx.read_graphml(GRAPHML)


@st.cache_data(show_spinner=False)
def load_cache():
    if not CACHE_FILE.exists():
        return None
    return json.loads(CACHE_FILE.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_region_countries(_G):
    """Return {region_key: [country, ...]} from graph node data."""
    mapping = {"WESTERN": [], "CENTRAL": [], "EASTERN": []}
    for node, data in _G.nodes(data=True):
        if data.get("node_type") == "Country":
            key = data.get("region", "").replace(" Region", "").upper()
            if key in mapping:
                mapping[key].append(node)
    return {k: sorted(v) for k, v in mapping.items()}


# ── pyvis graph builder ───────────────────────────────────────────────────────

def node_label(nid, ntype):
    if ntype == "Bulletin":
        return "B" + nid.replace("Bulletin_", "")
    if ntype == "Situation":
        parts = nid.replace("Situation_", "").split("_")
        bnum  = parts[0] if parts else "?"
        reg   = parts[1][0] if len(parts) > 1 else "?"
        return f"S{bnum}{reg}"
    if ntype == "Forecast":
        return "F" + nid.replace("Forecast_", "")
    return nid  # Region and Country use their name directly


def node_tooltip(nid, data):
    nt = data.get("node_type", "")
    if nt == "Bulletin":
        return (f"Bulletin {data.get('bulletin_number')}\n"
                f"{data.get('coverage_period','')}\n"
                f"Issued: {data.get('issue_date','')}")
    if nt == "Situation":
        return (f"Alert: {data.get('alert_level','')}\n"
                f"Region: {data.get('region','')}\n"
                f"{data.get('coverage_period','')}")
    if nt == "Forecast":
        return f"Forecast: {data.get('forecast_period','')}"
    if nt == "Country":
        return f"{nid}\nRegion: {data.get('region','')}"
    return nid

#Generate HTML to produce graph visualization
#Streamlit embeds it as an iframe
#Takes care of filters, hides or highlights nodes and edges
def build_pyvis(G, trav_nodes, trav_edges,
                alert_filter, node_type_filter, edge_type_filter,
                region_opt, country_opt, path_only=False,
                year_filter="All", month_filter=None,
                bul_min=532, bul_max=572, height="500px"):
    trav_nodes = set(trav_nodes or [])
    trav_edges = {tuple(e) for e in (trav_edges or [])}
    nodes_with_trav_edge = {n for e in trav_edges for n in e}

    # Pre-compute which bulletin numbers survive range/year/month filters
    # — used to cascade-hide Situation and Forecast nodes whose parent is hidden
    visible_bul_nums = set()
    for _nid, _d in G.nodes(data=True):
        if _d.get("node_type") != "Bulletin":
            continue
        _bnum = int(_d.get("bulletin_number", 0))
        if not (bul_min <= _bnum <= bul_max):
            continue
        if year_filter != "All" and int(_d.get("year", 0)) != int(year_filter):
            continue
        if month_filter and _d.get("month") not in month_filter:
            continue
        visible_bul_nums.add(_bnum)

    # Pre-compute region cascade from selected countries
    if country_opt:
        sel_regions = set()
        for cid in country_opt:
            r = G.nodes[cid].get("region", "").upper() if cid in G.nodes else ""
            if r:
                sel_regions.add(r)
    else:
        sel_regions = None

    net = Network(height=height, width="100%",
                  bgcolor="#1a1a2e", font_color="#cccccc",
                  directed=True)
    net.set_options("""{
      "physics": {
        "stabilization": {"iterations": 120, "fit": true},
        "barnesHut": {"gravitationalConstant": -6000, "springLength": 120}
      },
      "edges": {"smooth": {"type": "curvedCW", "roundness": 0.12}},
      "interaction": {"hover": true, "tooltipDelay": 80, "navigationButtons": true}
    }""")

    hidden_nodes = set()
    vis_nodes    = 0
    vis_edges    = 0

    for nid, data in G.nodes(data=True):
        ntype = str(data.get("node_type", "")).strip()

        if ntype not in node_type_filter:
            hidden_nodes.add(nid)
            continue

        if ntype == "Situation":
            al = data.get("alert_level", "")
            if al not in alert_filter:
                hidden_nodes.add(nid)
                continue

        # Cascade bulletin range/year/month to Situation and Forecast nodes
        if ntype in ("Situation", "Forecast"):
            if int(data.get("bulletin_number", 0)) not in visible_bul_nums:
                hidden_nodes.add(nid)
                continue

        # Bulletin-level filters: range, year, month
        if ntype == "Bulletin":
            bnum = int(data.get("bulletin_number", 0))
            if not (bul_min <= bnum <= bul_max):
                hidden_nodes.add(nid)
                continue
            if year_filter != "All" and int(data.get("year", 0)) != int(year_filter):
                hidden_nodes.add(nid)
                continue
            if month_filter and data.get("month") not in month_filter:
                hidden_nodes.add(nid)
                continue

        # Region filter: hide Country, Situation, Region nodes outside selected region
        if region_opt != "All":
            node_region = data.get("region", "")
            if ntype == "Country" and region_opt.upper() not in node_region.upper():
                hidden_nodes.add(nid)
                continue
            if ntype == "Situation" and region_opt.upper() not in node_region.upper():
                hidden_nodes.add(nid)
                continue
            if ntype == "Region" and region_opt.upper() not in nid.upper():
                hidden_nodes.add(nid)
                continue

        # Country filter: hide non-selected Country nodes
        if country_opt and ntype == "Country" and nid not in country_opt:
            hidden_nodes.add(nid)
            continue

        # Cascade: hide Region and Situation nodes outside selected countries' region
        if sel_regions:
            if ntype == "Region":
                if not any(r in nid.upper() for r in sel_regions):
                    hidden_nodes.add(nid)
                    continue
            if ntype == "Situation":
                node_region = data.get("region", "").upper()
                if not any(r in node_region for r in sel_regions):
                    hidden_nodes.add(nid)
                    continue

        is_active = nid in trav_nodes

        # Path-only mode: hide everything when no traversal, else hide non-traversed
        if path_only and (not nodes_with_trav_edge or nid not in nodes_with_trav_edge):
            hidden_nodes.add(nid)
            continue

        # Type color preserved — traversal shown via border+size only
        if ntype == "Bulletin":
            node_color = "#808080"
        elif ntype == "Region":
            node_color = "#6B2FA0"
        elif ntype == "Country":
            node_color = "#C9A8E0"
        elif ntype == "Forecast":
            node_color = "#00BCD4"
        elif ntype == "Situation":
            node_color = SITUATION_COLOR.get(data.get("alert_level", ""), "#888888")
        else:
            node_color = "#808080"

        vis_nodes += 1
        shape = "triangle" if ntype == "Situation" else "dot"
        net.add_node(
            nid,
            label=node_label(nid, ntype),
            color={"background": node_color,
                   "border": GOLDENROD if is_active else node_color,
                   "highlight": {"background": node_color, "border": GOLDENROD}},
            size=22 if is_active else 16,
            shape=shape,
            title=node_tooltip(nid, data),
            font={"size": 9, "color": "#cccccc"},
            borderWidth=5 if is_active else 1,
        )

    for u, v, edata in G.edges(data=True):
        if u in hidden_nodes or v in hidden_nodes:
            continue

        etype     = str(edata.get("edge_type", "")).strip()
        is_active = (u, v) in trav_edges

        if path_only and trav_edges and not is_active:
            continue

        if etype not in edge_type_filter and not is_active:
            continue

        if etype == "mentions":
            ha = float(edata.get("ha_treated", 0) or 0)
            if ha == 0 and not is_active:
                continue

        vis_edges += 1
        ha_val    = edata.get("ha_treated", "")
        hover_tip = etype + (f" | {ha_val} ha" if ha_val else "")

        net.add_edge(
            u, v,
            color=EDGE_ACTV if is_active else EDGE_COLOR.get(etype, "#555555"),
            width=3 if is_active else 0.7,
            title=hover_tip,   # ha_treated visible on hover only — no edge label
            arrows="to",
        )

    html = net.generate_html()
    # Make navigation arrows visible on dark background
    nav_css = (
        "<style>"
        ".vis-navigation .vis-button {"
        "  background-color:rgba(218,165,32,0.55) !important;"
        "  border-radius:4px !important; }"
        ".vis-navigation .vis-button:hover {"
        "  background-color:rgba(218,165,32,0.9) !important; }"
        "</style>"
    )
    return html.replace("</head>", nav_css + "</head>"), vis_nodes, vis_edges


# ── citation card ─────────────────────────────────────────────────────────────
#RAG chunk citation card
def citation_card(chunk):
    score  = chunk["score"]
    bnum   = chunk["bulletin_number"]
    page   = chunk["page"]
    cov    = chunk.get("coverage_period", "")
    snip   = chunk["snippet"][:320].replace("<", "&lt;").replace(">", "&gt;")
    bar_w  = int(score * 100)
    return f"""
<div style="border:1px solid #333;border-radius:8px;padding:12px;margin:8px 0;
            background:#16213e;font-family:sans-serif;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;">
    <span style="font-weight:600;color:#c9a8e0;">
      Bulletin No.&nbsp;{bnum} &middot; Page {page}
    </span>
  </div>
  <div style="font-size:12px;color:#888;margin:3px 0;">{cov}</div>
  <div style="background:#333;border-radius:3px;height:4px;margin:6px 0;">
    <div style="background:{GOLDENROD};width:{bar_w}%;height:4px;border-radius:3px;"></div>
  </div>
  <div style="font-size:11px;color:#aaa;margin-bottom:6px;">Similarity: {score:.3f}</div>
  <div style="font-size:13px;color:#ddd;line-height:1.5;">{snip}&hellip;</div>
</div>
"""


# ── traversal result renderers ────────────────────────────────────────────────

def render_treatment_yoy(graphrag):
    year_totals  = graphrag.get("year_totals", {})
    ranked_by_yr = graphrag.get("ranked_by_year", {})
    for yr_str, ranked in ranked_by_yr.items():
        total = year_totals.get(yr_str, 0)
        st.markdown(f"**{yr_str}  —  {total:,.0f} ha total**")
        rows = [
            f"| {r['country']} | {r['ha']:,.0f} | "
            f"{', '.join('B'+str(b) for b in r['bulletins'])} |"
            for r in ranked[:8]
        ]
        if rows:
            st.markdown(
                "| Country | ha treated | Bulletins |\n"
                "|---|---|---|\n" + "\n".join(rows)
            )


def render_alert_regions(graphrag):
    from collections import Counter
    alerts_by_region = graphrag.get("alerts_by_region", {})
    for region, records in alerts_by_region.items():
        st.markdown(f"**{region} Region**")
        counts = Counter(r["alert"] for r in records)
        rows = [f"| {lvl} | {cnt} |"
                for lvl, cnt in sorted(counts.items())]
        st.markdown("| Alert level | Bulletins |\n|---|---|\n" + "\n".join(rows))


def render_country_mentions(graphrag):
    country    = graphrag.get("country", "")
    total_ha   = graphrag.get("total_ha", 0)
    data_pts   = graphrag.get("data_points", [])
    treated    = [d for d in data_pts if d.get("ha_treated", 0) > 0]

    st.markdown(
        f"**{country}** — {total_ha:,.0f} ha treated "
        f"in {len(treated)} of {len(data_pts)} bulletins"
    )
    if treated:
        rows = [
            f"| B{d['bulletin']} | {d['coverage']} | {d['ha_treated']:,.0f} |"
            for d in treated
        ]
        st.markdown(
            "| Bulletin | Period | ha treated |\n|---|---|---|\n"
            + "\n".join(rows)
        )
    else:
        st.caption("No treatment operations recorded for this country.")


# ── page layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Desert Locust Bulletin Dashboard",
    layout="wide",
    page_icon="🌿",
)

# global style tweaks
st.markdown("""
<style>
  .block-container { padding-top: 2.5rem !important; padding-bottom: 1rem !important; }
  section[data-testid="stSidebar"] > div:first-child { padding-top: 0.4rem !important; }
  .streamlit-expanderHeader { padding: 0.35rem 0.6rem !important; font-size: 0.82rem; }
  input[type=number] { padding: 0.2rem 0.4rem !important; }
  /* multiselect selected tag styling */
  span[data-baseweb="tag"] {
      background-color: #1e3358 !important;
      border: 1px solid #6B2FA0 !important;
      border-radius: 4px !important;
  }
  span[data-baseweb="tag"] span { color: #c9a8e0 !important; font-size: 11px !important; }
  /* tighten question selector area */
  hr { margin: 0.25rem 0 !important; }
  div[data-testid="stSelectbox"] { margin-top: -0.3rem !important; margin-bottom: -0.4rem !important; }
</style>
""", unsafe_allow_html=True)

# header: title + tagline full width; legend tucked to the right
title_col, legend_col = st.columns([7, 1])
with title_col:
    st.subheader("Desert Locust Bulletin Dashboard")
    st.caption(f"[FAO Desert Locust Bulletins No. 532-572 · 2023-2026]({FAO_URL})")
with legend_col:
    with st.expander("Legend", expanded=False):
        entries = [
            ("Bulletin",  TYPE_COLOR["Bulletin"],  "circle"),
            ("Region",    TYPE_COLOR["Region"],    "circle"),
            ("Country",   TYPE_COLOR["Country"],   "circle"),
            ("Forecast",  TYPE_COLOR["Forecast"],  "circle"),
            ("CALM",      SITUATION_COLOR["CALM"],    "triangle"),
            ("CAUTION",   SITUATION_COLOR["CAUTION"], "triangle"),
            ("WARNING",   SITUATION_COLOR["WARNING"], "triangle"),
            ("ALERT",     SITUATION_COLOR["ALERT"],   "triangle"),
            ("CRISIS",    SITUATION_COLOR["CRISIS"],  "triangle"),
            ("Traversal", GOLDENROD,                  "circle"),
        ]
        for label, color, shape in entries:
            if shape == "triangle":
                icon = (
                    f'<div style="width:0;height:0;'
                    f'border-left:6px solid transparent;'
                    f'border-right:6px solid transparent;'
                    f'border-bottom:12px solid {color};'
                    f'flex-shrink:0;margin-top:1px;"></div>'
                )
            else:
                icon = (
                    f'<div style="background:{color};width:12px;height:12px;'
                    f'border-radius:50%;flex-shrink:0;"></div>'
                )
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:5px;margin:3px 0;">'
                f'{icon}<span style="font-size:10px;">{label}</span></div>',
                unsafe_allow_html=True,
            )

st.divider()

# load data early so sidebar can use graph for country filtering
G     = load_graph()
cache = load_cache()
region_countries = get_region_countries(G)

# sidebar — three collapsible accordion sections
with st.sidebar:
    f_col, r_col = st.columns([2, 1])
    f_col.markdown(
        '<p style="font-size:0.95rem;font-weight:600;margin:0 0 0.4rem 0;">Filters</p>',
        unsafe_allow_html=True,
    )
    if r_col.button("Reset", use_container_width=True):
        for k, v in FILTER_DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    with st.expander("Query scope", expanded=True):
        yc, mc = st.columns(2)
        year_opt  = yc.selectbox("Year",  ["All", 2023, 2024, 2025, 2026], key="year_sel")
        month_opt = mc.multiselect("Month", MONTHS, key="month_sel")

        st.markdown(
            '<p style="font-size:0.8rem;margin:0.4rem 0 0.1rem 0;color:#888;">Bulletin</p>',
            unsafe_allow_html=True,
        )
        b1, mid, b2 = st.columns([5, 1, 5])
        bul_min = b1.number_input("From", min_value=532, max_value=572, value=532,
                                  label_visibility="hidden", key="bul_min")
        mid.markdown(
            '<div style="text-align:center;padding-top:0.35rem;font-size:0.8rem;'
            'color:#aaa;">to</div>',
            unsafe_allow_html=True,
        )
        bul_max = b2.number_input("To", min_value=532, max_value=572, value=572,
                                  label_visibility="hidden", key="bul_max")

        region_opt = st.selectbox("Region", ["All", "WESTERN", "CENTRAL", "EASTERN"],
                                  key="region_sel")

        if region_opt == "All":
            country_choices = sorted(KNOWN_COUNTRIES)
        else:
            country_choices = region_countries.get(region_opt, sorted(KNOWN_COUNTRIES))
        country_opt = st.multiselect("Country", country_choices, key="country_sel")

    with st.expander("Graph display", expanded=True):
        st.markdown(
            '<p style="font-size:0.8rem;margin:0 0 0.3rem 0;color:#aaa;">Node types</p>',
            unsafe_allow_html=True,
        )
        node_type_filter = []
        for ntype, color in TYPE_COLOR.items():
            if ntype == "Situation":
                continue
            dot_col, chk_col = st.columns([2, 5])
            dot_col.markdown(
                f'<div style="background:{color};width:14px;height:14px;'
                f'border-radius:50%;margin-top:7px;"></div>',
                unsafe_allow_html=True,
            )
            if chk_col.checkbox(ntype, value=True, key=f"nt_{ntype}"):
                node_type_filter.append(ntype)

        st.markdown(
            '<p style="font-size:0.8rem;margin:0.5rem 0 0.2rem 0;color:#aaa;">Traversal</p>',
            unsafe_allow_html=True,
        )
        path_only = st.checkbox("Path only (hide non-traversed)", value=False,
                                key="path_only")

        st.markdown(
            '<p style="font-size:0.8rem;margin:0.5rem 0 0.3rem 0;color:#aaa;">Alert levels</p>',
            unsafe_allow_html=True,
        )
        alert_filter = []
        for level in ALL_ALERT_LEVELS:
            color = SITUATION_COLOR[level]
            dot_col, chk_col = st.columns([2, 5])
            dot_col.markdown(
                f'<div style="width:0;height:0;'
                f'border-left:7px solid transparent;'
                f'border-right:7px solid transparent;'
                f'border-bottom:14px solid {color};'
                f'margin-top:8px;"></div>',
                unsafe_allow_html=True,
            )
            if chk_col.checkbox(level, value=True, key=f"al_{level}"):
                alert_filter.append(level)
        if any(alert_filter):
            node_type_filter.append("Situation")

        st.markdown(
            '<p style="font-size:0.8rem;margin:0.5rem 0 0.3rem 0;color:#aaa;">Edge types</p>',
            unsafe_allow_html=True,
        )
        edge_type_filter = []
        for etype in ALL_EDGE_TYPES:
            if st.checkbox(etype.replace("_", " "), value=True, key=f"et_{etype}"):
                edge_type_filter.append(etype)


    with st.expander("RAG tuning", expanded=False):
        top_k     = st.slider("Retrieved pages limit", 1, 20, 4, key="top_k_sl")
        min_score = st.slider("Minimum score", 0.0, 1.0, 0.65, step=0.01,
                              key="min_score_sl")

if cache is None:
    st.error(
        "Demo cache not found. Run the pipeline first:\n\n"
        "```\npython pipeline/build_demo.py\n```"
    )
    st.stop()

# ── query selector — only the four showcase cases ─────────────────────────────
VISIBLE_IDS   = {"qMor", "q4", "q1", "q3"}        # parked: all others
DISPLAY_ORDER = ["qMor", "q4", "q1", "q3"]
visible_qs = sorted(
    [q for q in cache["queries"] if q["id"] in VISIBLE_IDS],
    key=lambda q: DISPLAY_ORDER.index(q["id"]),
)
q_options = [None] + visible_qs                    # None = blank "explore" slot
q_idx = st.selectbox(
    "Select a question",
    range(len(q_options)),
    format_func=lambda i: (
        "— Explore the graph —" if q_options[i] is None
        else q_options[i]["question"]
    ),
)
q_data = q_options[q_idx]

st.divider()

# resolve traversal data (empty when blank is selected)
if q_data is not None:
    graphrag   = q_data["graphrag"]
    trav_nodes = graphrag.get("trav_nodes", [])
    trav_edges = graphrag.get("trav_edges", [])
else:
    graphrag   = None
    trav_nodes = []
    trav_edges = []

# two columns
col_rag, col_graph = st.columns(2)

# ── left: Document Retrieval ──────────────────────────────────────────────────
with col_rag:
    st.subheader("Document Retrieval")

    if q_data is None:
        st.caption("Select a question above to see retrieved bulletin pages.")
    else:
        rag    = q_data["rag"]
        chunks = rag["chunks"]

        filtered = [
            c for c in chunks
            if c["score"] >= min_score
            and (bul_min <= c["bulletin_number"] <= bul_max)
            and (not country_opt or any(
                ct.lower() in c["snippet"].lower() for ct in country_opt))
        ][:top_k]

        if rag.get("summary"):
            st.markdown(clean_summary(rag["summary"]))
            st.divider()

        if not filtered:
            st.info("No match. Try RAG tuning → Minimum score.")
        else:
            st.caption(f"Showing {len(filtered)} of {len(chunks)} retrieved pages")
            for chunk in filtered:
                st.markdown(citation_card(chunk), unsafe_allow_html=True)

# ── right: Knowledge Graph Traversal ─────────────────────────────────────────
with col_graph:
    st.subheader("Knowledge Graph Traversal")

    graph_html, vis_nodes, vis_edges = build_pyvis(
        G, trav_nodes, trav_edges,
        alert_filter=alert_filter,
        node_type_filter=node_type_filter,
        edge_type_filter=edge_type_filter,
        region_opt=region_opt,
        country_opt=country_opt,
        path_only=path_only,
        year_filter=year_opt,
        month_filter=month_opt if month_opt else None,
        bul_min=bul_min,
        bul_max=bul_max,
        height="500px",
    )
    if graphrag is not None:
        tv_path  = graphrag.get("traversal_path", "")
        tv_nodes = graphrag.get("nodes_visited", 0)
        tv_edges = graphrag.get("edges_traversed", 0)
        if graphrag["has_results"]:
            st.info(f"{tv_path}  |  {tv_nodes} traversal nodes / {tv_edges} edges  ·  {vis_nodes} visible")
        else:
            st.caption(f"No structured traversal path for this query  ·  {vis_nodes} nodes visible")
    else:
        st.info(f"Graph: {vis_nodes} nodes  ·  {vis_edges} edges")

    components.html(graph_html, height=520, scrolling=False)

    if graphrag is not None:
        st.markdown(clean_summary(graphrag.get("summary", "")))
        if graphrag["has_results"]:
            with st.expander("Traversal results", expanded=True):
                gtype = graphrag.get("type", "")
                if gtype == "treatment_yoy":
                    render_treatment_yoy(graphrag)
                elif gtype == "alert_multi_region":
                    render_alert_regions(graphrag)
                elif gtype == "country_mentions":
                    render_country_mentions(graphrag)
