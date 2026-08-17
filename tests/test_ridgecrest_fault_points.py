import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ridgecrest_fault_points import build_fault_sampling_points


def test_builds_paired_points_from_independent_fault_geometry(tmp_path):
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "Label": "Paxton Ranch Fault Zone",
                    "IdentityConfidence": "certain",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [-117.70, 35.60],
                        [-117.60, 35.70],
                        [-117.50, 35.80],
                        [-117.40, 35.90],
                    ],
                },
            }
        ],
    }
    path = tmp_path / "fault.geojson"
    path.write_text(json.dumps(feature_collection), encoding="utf-8")

    nodes, points, _ = build_fault_sampling_points(
        path,
        node_spacing_km=5.0,
        side_offset_km=2.0,
    )

    assert len(nodes) >= 3
    assert len(points) == 2 * len(nodes)
    assert set(points["side"]) == {"minus", "plus"}
    assert np.allclose(np.abs(points["fault_offset_km"]), 2.0)
