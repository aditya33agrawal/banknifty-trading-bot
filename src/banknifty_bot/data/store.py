from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .cleaner import coverage_summary


def _partition_path(root: Path, symbol: str, interval: str, year: int) -> Path:
    safe_symbol = symbol.replace("^", "").replace("/", "_")
    return root / f"symbol={safe_symbol}" / f"interval={interval}" / f"year={year}"


def write_partitioned(df: pd.DataFrame, root_dir: str | Path, symbol: str, interval: str) -> dict:
    """Write df to year-partitioned parquet, merging with any existing data for
    that partition, and update a coverage manifest.
    """
    root = Path(root_dir)
    manifest: dict = {}
    if df.empty:
        return manifest

    for year, year_df in df.groupby(df.index.year):
        part_dir = _partition_path(root, symbol, interval, int(year))
        part_dir.mkdir(parents=True, exist_ok=True)
        part_file = part_dir / "data.parquet"

        if part_file.exists():
            existing = pd.read_parquet(part_file)
            combined = pd.concat([existing, year_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = year_df.sort_index()

        combined.to_parquet(part_file)
        manifest[str(year)] = coverage_summary(combined)

    _update_manifest(root, symbol, interval, manifest)
    return manifest


def _update_manifest(root: Path, symbol: str, interval: str, year_coverage: dict) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    key = f"{symbol}|{interval}"
    manifest.setdefault(key, {}).update(year_coverage)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def read_partitioned(
    root_dir: str | Path, symbol: str, interval: str, start: str, end: str
) -> pd.DataFrame:
    root = Path(root_dir)
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)

    frames = []
    for year in years:
        part_file = _partition_path(root, symbol, interval, year) / "data.parquet"
        if part_file.exists():
            frames.append(pd.read_parquet(part_file))

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[(df.index >= pd.Timestamp(start, tz=df.index.tz)) &
                  (df.index <= pd.Timestamp(end, tz=df.index.tz))]
