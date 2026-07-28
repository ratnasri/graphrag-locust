"""
Extract structured entities from individual bulletin text files.
Produces: data/graph/entities.jsonl — one JSON record per bulletin.

Entities extracted:
  - Bulletin metadata (number, issue date, coverage period, forecast period)
  - Regional alert levels (Western / Central / Eastern)
  - Country-level situations and forecasts
  - Locust activities (ha treated, hopper/adult/swarm mentions)
"""

import re
import json
from pathlib import Path

INPUT_DIR = Path("G:/FAOLocustKG/data/text/individual")
OUTPUT_DIR = Path("G:/FAOLocustKG/data/graph")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "entities.jsonl"

REGIONS = ["WESTERN", "CENTRAL", "EASTERN"]
ALERT_LEVELS = ["CALM", "CAUTION", "WARNING", "ALERT", "CRISIS"]

KNOWN_COUNTRIES = [
    "Morocco", "Algeria", "Tunisia", "Libya", "Mauritania",
    "Western Sahara", "Mali", "Niger", "Chad", "Senegal",
    "Egypt", "Sudan", "Eritrea", "Ethiopia", "Djibouti", "Somalia",
    "Saudi Arabia", "Yemen", "Oman", "Iran", "Pakistan", "India",
    "Kenya", "Uganda", "Tanzania", "Mozambique", "Afghanistan",
    "Iraq", "Syria", "Jordan", "Kuwait", "UAE",
]


# --- Bulletin metadata ---

def extract_bulletin_number(text):
    m = re.search(r"No\.\s*(\d+)", text)
    return int(m.group(1)) if m else None


def extract_issue_date(text):
    m = re.search(
        r"No\.\s*\d+\s+(\d{1,2}\s+\w+\s+\d{4})",
        text
    )
    return m.group(1).strip() if m else None


def extract_coverage_period(text):
    m = re.search(r"General situation during\s+(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else None


def extract_forecast_period(text):
    m = re.search(r"Forecast until\s+(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else None


# --- Regional alert levels ---

def extract_regional_alerts(text):
    alerts = {}
    for region in REGIONS:
        pattern = re.compile(
            rf"{region}\s+REGION\s*:\s*({'|'.join(ALERT_LEVELS)})",
            re.IGNORECASE
        )
        m = pattern.search(text)
        if m:
            alerts[region] = m.group(1).upper()
    return alerts


# --- Country mentions ---

def extract_country_sections(text):
    """
    Extract per-country situation and forecast text.
    Looks for bullet/section patterns like:
      Country\n• situation\n...\n• forecast\n...
    """
    country_data = {}

    for country in KNOWN_COUNTRIES:
        pattern = re.compile(
            rf"\b{re.escape(country)}\b(.{{0,2000}}?)(?={'|'.join(KNOWN_COUNTRIES)}|\Z)",
            re.DOTALL | re.IGNORECASE
        )
        for m in pattern.finditer(text):
            block = m.group(0).strip()
            if len(block) < 30:
                continue

            situation = ""
            forecast = ""

            sit_m = re.search(
                r"[•\-]?\s*situation[:\s]*(.+?)(?=[•\-]?\s*forecast|$)",
                block, re.DOTALL | re.IGNORECASE
            )
            fore_m = re.search(
                r"[•\-]?\s*forecast[:\s]*(.+?)$",
                block, re.DOTALL | re.IGNORECASE
            )

            if sit_m:
                situation = re.sub(r"\s+", " ", sit_m.group(1)).strip()[:500]
            if fore_m:
                forecast = re.sub(r"\s+", " ", fore_m.group(1)).strip()[:500]

            if situation or forecast:
                country_data[country] = {
                    "situation": situation,
                    "forecast": forecast,
                }
                break

    return country_data


# --- Locust activities ---

def extract_ha_treated(text):
    """Extract hectares treated per country."""
    ha_data = {}
    pattern = re.compile(
        r"(\w[\w\s]{1,30}?)\s*[\n\t]+\s*(\d[\d,\.]*)\s*ha",
        re.IGNORECASE
    )
    for m in pattern.finditer(text):
        country_candidate = m.group(1).strip()
        ha = m.group(2).replace(",", "")
        for country in KNOWN_COUNTRIES:
            if country.lower() in country_candidate.lower():
                ha_data[country] = float(ha)
                break
    return ha_data


def extract_locust_stages(text):
    """Detect mentions of locust life stages and swarm types."""
    stages = {}
    patterns = {
        "hoppers": r"\bhoppers?\b",
        "adults": r"\badults?\b",
        "swarms": r"\bswarms?\b",
        "bands": r"\bbands?\b",
        "groups": r"\bgroups?\b",
        "solitarious": r"\bsolitarious\b",
        "gregarious": r"\bgregarious\b",
    }
    for stage, pattern in patterns.items():
        count = len(re.findall(pattern, text, re.IGNORECASE))
        if count > 0:
            stages[stage] = count
    return stages


# --- Main extraction ---

def extract_bulletin(txt_path):
    text = txt_path.read_text(encoding="utf-8")

    bulletin_no = extract_bulletin_number(text)
    issue_date = extract_issue_date(text)
    coverage = extract_coverage_period(text)
    forecast_period = extract_forecast_period(text)
    regional_alerts = extract_regional_alerts(text)
    countries = extract_country_sections(text)
    ha_treated = extract_ha_treated(text)
    locust_stages = extract_locust_stages(text)

    countries_mentioned = [c for c in KNOWN_COUNTRIES
                           if re.search(rf"\b{re.escape(c)}\b", text, re.IGNORECASE)]

    return {
        "source_file": txt_path.name,
        "bulletin_number": bulletin_no,
        "issue_date": issue_date,
        "coverage_period": coverage,
        "forecast_period": forecast_period,
        "regional_alerts": regional_alerts,
        "countries_mentioned": countries_mentioned,
        "country_detail": countries,
        "ha_treated": ha_treated,
        "locust_stages": locust_stages,
    }


def run():
    txt_files = sorted(INPUT_DIR.glob("bulletin_*.txt"))
    print(f"Processing {len(txt_files)} bulletins...\n")

    records = []
    for txt_path in txt_files:
        record = extract_bulletin(txt_path)
        records.append(record)
        alerts = record["regional_alerts"]
        countries_n = len(record["countries_mentioned"])
        ha = sum(record["ha_treated"].values())
        print(
            f"  No. {record['bulletin_number']:>3}  {record['coverage_period'] or '?':30s}"
            f"  alerts={alerts}  countries={countries_n}  ha={ha:.0f}"
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(records)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
