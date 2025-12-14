"""
Compute CIgreen (Chlorophyll Index Green) from separate Sentinel-2 band rasters (B03 green, B08 NIR)
stored under map/<year>/ with pattern S2_<yyyymmdd>_<BXX>_10m.tif.

CIgreen = (NIR / Green) - 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import rasterio


BANDS = {"B03": "green", "B08": "nir"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CIgreen from separate Sentinel-2 band rasters.")
    parser.add_argument("--year", type=int, default=2017, help="Year folder under map/. Default: 2017.")
    parser.add_argument(
        "--date",
        default=None,
        help="Optional date filter (YYYYMMDD). If not provided, uses the first matching set in the year folder.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CIgreen GeoTIFF path. Default: map/<year>/cigreen_<yyyymmdd>.tif based on matched date.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1 / 10000.0,
        help="Scale factor to convert DN to reflectance. Default: 1/10000 for L2A. Use 1.0 if already scaled.",
    )
    return parser.parse_args()


def find_band_paths(year: int, date_filter: Optional[str]) -> (Dict[str, Path], str):
    base = Path("map") / str(year)
    if not base.exists():
        raise SystemExit(f"Year folder not found: {base}")

    matches = {key: [] for key in BANDS}
    for band in BANDS:
        for p in base.glob(f"S2_*_{band}_10m.tif"):
            parts = p.stem.split("_")
            if len(parts) < 3:
                continue
            date_str = parts[1]
            if date_filter and date_str != date_filter:
                continue
            matches[band].append((date_str, p))

    dates = None
    for band, items in matches.items():
        if not items:
            raise SystemExit(f"No files found for band {band} in {base} with given filters.")
        band_dates = {d for d, _ in items}
        dates = band_dates if dates is None else dates & band_dates
    if not dates:
        raise SystemExit("No common date across bands. Specify --date to pick one.")
    date_str = sorted(dates)[0]
    paths = {}
    for band, items in matches.items():
        candidates = [p for d, p in items if d == date_str]
        paths[band] = candidates[0]
    return paths, date_str


def compute_cigreen(green: np.ndarray, nir: np.ndarray, nodata_mask: np.ndarray | None = None, scale: float = 1.0) -> np.ndarray:
    green = green.astype("float32") * scale
    nir = nir.astype("float32") * scale
    invalid_ref = (green <= 0) | (nir < 0) | (green > 1.5) | (nir > 1.5)
    denom = green
    bad_denom = np.isclose(denom, 0, atol=1e-6) | (denom <= 0)
    mask = invalid_ref | bad_denom
    if nodata_mask is not None:
        mask = mask | nodata_mask
    cigreen = np.empty_like(green, dtype="float32")
    valid = ~mask
    cigreen[valid] = (nir[valid] / green[valid]) - 1
    cigreen[mask] = np.nan
    # clip implausible extremes
    implausible = (cigreen < -5) | (cigreen > 5)
    cigreen[implausible] = np.nan
    return cigreen


def main() -> None:
    args = parse_args()
    band_paths, date_str = find_band_paths(args.year, args.date)
    green_path = band_paths["B03"]
    nir_path = band_paths["B08"]

    out_path: Path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("map") / str(args.year) / f"cigreen_{date_str}.tif"

    with rasterio.open(green_path) as green_src, rasterio.open(nir_path) as nir_src:
        if not (green_src.crs == nir_src.crs and green_src.transform == nir_src.transform):
            raise SystemExit("Input rasters differ in CRS or transform; align them first.")

        profile = green_src.profile.copy()
        profile.update(
            {
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
            }
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in green_src.block_windows(1):
                green = green_src.read(1, window=window)
                nir = nir_src.read(1, window=window)
                masks = []
                for band, nod in zip((green, nir), (green_src.nodata, nir_src.nodata)):
                    if nod is not None:
                        masks.append(band == nod)
                    masks.append((band <= 0) | (band > 12000))
                nodata_mask = np.logical_or.reduce(masks) if masks else None
                cigreen_block = compute_cigreen(green, nir, nodata_mask, scale=args.scale)
                dst.write(cigreen_block.astype("float32"), 1, window=window)

    print(f"Saved CIgreen to {out_path} (year {args.year}, date {date_str}, scale {args.scale})")


if __name__ == "__main__":
    main()
