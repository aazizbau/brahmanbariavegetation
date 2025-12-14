"""
Compute NDVI from separate Sentinel-2 band rasters (B04 red and B08 NIR)
and write a GeoTIFF.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute NDVI from separate B04 and B08 rasters.")
    parser.add_argument("--year", type=int, default=2017, help="Year folder under map/. Default: 2017.")
    parser.add_argument(
        "--date",
        default=None,
        help="Optional date filter (YYYYMMDD). If not provided, uses the first matching pair in the year folder.",
    )
    parser.add_argument(
        "--red",
        default=None,
        help="Optional explicit path to red band raster (B04). Overrides year/date search.",
    )
    parser.add_argument(
        "--nir",
        default=None,
        help="Optional explicit path to NIR band raster (B08). Overrides year/date search.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output NDVI GeoTIFF path. Default: map/<year>/ndvi_<yyyymmdd>.tif based on matched date.",
    )
    return parser.parse_args()


def compute_ndvi(red: np.ndarray, nir: np.ndarray, nodata_mask: np.ndarray | None = None) -> np.ndarray:
    red = red.astype("float32")
    nir = nir.astype("float32")
    denom = nir + red
    zero_denom = denom == 0
    mask = zero_denom if nodata_mask is None else (zero_denom | nodata_mask)
    ndvi = np.empty_like(red, dtype="float32")
    valid = ~mask
    ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
    ndvi[mask] = np.nan
    return ndvi


def find_band_pair(year: int, date_filter: Optional[str]) -> Tuple[Path, Path, str]:
    base = Path("map") / str(year)
    if not base.exists():
        raise SystemExit(f"Year folder not found: {base}")
    red_files = sorted(base.glob("S2_*_B04_10m.tif"))
    if not red_files:
        raise SystemExit(f"No red band files matching S2_*_B04_10m.tif in {base}")
    pairs = []
    for red_path in red_files:
        stem = red_path.stem  # e.g., S2_20171103_B04_10m
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        date_str = parts[1]
        if date_filter and date_str != date_filter:
            continue
        nir_path = red_path.with_name(stem.replace("B04", "B08") + red_path.suffix)
        if nir_path.exists():
            pairs.append((red_path, nir_path, date_str))
    if not pairs:
        raise SystemExit("No matching red/NIR band pairs found with given filters.")
    if len(pairs) > 1 and date_filter is None:
        raise SystemExit(
            f"Multiple matching pairs found ({len(pairs)}). "
            "Specify --date YYYYMMDD to pick one."
        )
    return pairs[0]


def main() -> None:
    args = parse_args()
    if args.red and args.nir:
        red_path = Path(args.red)
        nir_path = Path(args.nir)
        date_str = None
    else:
        red_path, nir_path, date_str = find_band_pair(args.year, args.date)

    if not red_path.exists():
        raise SystemExit(f"Red band not found: {red_path}")
    if not nir_path.exists():
        raise SystemExit(f"NIR band not found: {nir_path}")

    out_path: Path
    if args.output:
        out_path = Path(args.output)
    else:
        if not date_str:
            raise SystemExit("Cannot infer output name without a matched date. Provide --output.")
        out_path = Path("map") / str(args.year) / f"ndvi_{date_str}.tif"

    with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
        if red_src.crs != nir_src.crs or red_src.transform != nir_src.transform:
            raise SystemExit("Red and NIR rasters differ in CRS or transform; align them first.")

        profile = red_src.profile.copy()
        profile.update(
            {
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
            }
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in red_src.block_windows(1):
                red = red_src.read(1, window=window)
                nir = nir_src.read(1, window=window)
                nodata_mask = None
                if red_src.nodata is not None:
                    nodata_mask = red == red_src.nodata
                ndvi_block = compute_ndvi(red, nir, nodata_mask)
                dst.write(ndvi_block.astype("float32"), 1, window=window)

    print(f"Saved NDVI to {out_path} (year {args.year}, date {args.date or date_str or 'unknown'})")


if __name__ == "__main__":
    main()
