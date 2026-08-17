"""Download public CGS 2019 Ridgecrest mapped fault ruptures as GeoJSON."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures.geojson"
METADATA = ROOT / "data" / "cgs_2019_ridgecrest_fault_ruptures_source.json"
SERVICE = (
    "https://gis.conservation.ca.gov/server/rest/services/CGS/"
    "2019_Ridgecrest_Earthquakes_Rupture_Mapping/FeatureServer/0/query"
)


def request_json(parameters: dict[str, object], attempts: int = 8) -> dict:
    url = f"{SERVICE}?{urlencode(parameters)}"
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=120) as response:
                return json.load(response)
        except Exception as caught:
            error = caught
            if attempt + 1 == attempts:
                break
            delay = min(2 ** attempt, 20)
            print(f"Request failed; retrying in {delay} s: {caught}")
            time.sleep(delay)
    raise RuntimeError(f"CGS request failed after {attempts} attempts") from error


parameters = {
    "where": "Type='fault'",
    "outFields": "OBJECTID,Type,ExistenceConfidence,IdentityConfidence,Label",
    "returnGeometry": "true",
    "outSR": 4326,
    "f": "geojson",
}
count_response = request_json(
    {
        "where": parameters["where"],
        "returnCountOnly": "true",
        "f": "json",
    }
)
expected = int(count_response["count"])
features: list[dict] = []
page_size = 2000
for offset in range(0, expected, page_size):
    page_parameters = {
        **parameters,
        "resultOffset": offset,
        "resultRecordCount": page_size,
        "orderByFields": "OBJECTID",
    }
    page = request_json(page_parameters)
    if "error" in page:
        raise RuntimeError(page["error"])
    features.extend(page.get("features", []))
    print(f"Downloaded {len(features):,}/{expected:,}")

if len(features) != expected:
    raise RuntimeError(f"Expected {expected} features, received {len(features)}")

collection = {
    "type": "FeatureCollection",
    "name": "CGS 2019 Ridgecrest mapped fault ruptures",
    "crs": {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
    },
    "features": features,
}
OUTPUT.write_text(json.dumps(collection), encoding="utf-8")
METADATA.write_text(
    json.dumps(
        {
            "source_service": SERVICE.rsplit("/query", 1)[0],
            "query": parameters,
            "feature_count": expected,
            "description": (
                "California Geological Survey mapped surface rupture features "
                "for the 2019 Ridgecrest earthquake sequence."
            ),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(OUTPUT)
