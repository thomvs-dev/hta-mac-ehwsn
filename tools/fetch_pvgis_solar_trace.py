"""Download and normalize a citable PVGIS hourly irradiance trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?"
    "lat=39.7423&lon=-105.1785&startyear=2020&endyear=2020&"
    "outputformat=json&components=0&pvcalculation=0&raddatabase=PVGIS-ERA5"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_hourly(payload: dict) -> list[tuple[str, float]]:
    rows = payload.get("outputs", {}).get("hourly", [])
    result = []
    for row in rows:
        timestamp = str(row.get("time", ""))
        irradiance = row.get("G(i)")
        if irradiance is None:
            components = [row.get(name, 0.0) for name in ("Gb(i)", "Gd(i)", "Gr(i)")]
            irradiance = sum(float(value or 0.0) for value in components)
        value = max(0.0, float(irradiance))
        if timestamp:
            result.append((timestamp, value))
    if len(result) < 24 * 300:
        raise RuntimeError(f"PVGIS response contains only {len(result)} hourly rows")
    if max(value for _, value in result) <= 0.0:
        raise RuntimeError("PVGIS response has no positive irradiance")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    request = urllib.request.Request(args.url, headers={"User-Agent": "HTA-MAC-research/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8"))
    rows = parse_hourly(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_utc", "global_in_plane_irradiance_w_m2"])
        writer.writerows(rows)
    trace_bytes = args.output.read_bytes()
    metadata = {
        "schema_version": 1,
        "status": "external_trace_downloaded_and_hashed",
        "source_url": args.url,
        "dataset_doi": "10.2905/JRC.FDTQ22G",
        "location": {"description": "NREL SRRL coordinates", "latitude": 39.7423, "longitude": -105.1785},
        "year": 2020,
        "rows": len(rows),
        "raw_response_sha256": sha256_bytes(raw),
        "trace_sha256": sha256_bytes(trace_bytes),
        "minimum_w_m2": min(value for _, value in rows),
        "maximum_w_m2": max(value for _, value in rows),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
