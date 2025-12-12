"""
Check whether all expected Sentinel-2 composite tiles for Brahmanbaria have
been downloaded to the local folder. Uses the same tiling scheme as
download_gee_s2_composite.py (tile size in km with overlap).
"""

from __future__ import annotations

import argparse
import calendar
import math
from pathlib import Path
from typing import Iterable, List, Tuple


def brahmanbaria_coords() -> list:
    return [
        [90.5024, 24.4451],  # upper left
        [90.6174, 23.5101],  # lower left
        [91.4834, 23.5049],  # lower right
        [91.4655, 24.3480],  # upper right
        [90.5024, 24.4451],  # close polygon
    ]


def parse_month(month: str) -> int:
    try:
        m = int(month)
        if 1 <= m <= 12:
            return m
    except ValueError:
        pass
    month_lower = month.strip().lower()
    for idx, name in enumerate(calendar.month_name):
        if name and name.lower().startswith(month_lower):
            return idx
    raise ValueError(f"Invalid month: {month}")


def bbox_from_coords(coords: List[List[float]]) -> Tuple[float, float, float, float]:
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


def meters_per_degree(mean_lat: float) -> Tuple[float, float]:
    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * math.cos(math.radians(mean_lat))
    return meters_per_deg_lat, meters_per_deg_lon


def split_bbox_by_km(
    bbox: Tuple[float, float, float, float], tile_w_km: float, tile_h_km: float, overlap_km: float
) -> Iterable[Tuple[Tuple[float, float, float, float], Tuple[int, int]]]:
    lon_min, lat_min, lon_max, lat_max = bbox
    mean_lat = (lat_min + lat_max) / 2
    m_per_deg_lat, m_per_deg_lon = meters_per_degree(mean_lat)

    if tile_w_km <= overlap_km or tile_h_km <= overlap_km:
        raise ValueError("Tile width/height must be greater than overlap.")

    tile_w_deg = (tile_w_km * 1000) / m_per_deg_lon
    tile_h_deg = (tile_h_km * 1000) / m_per_deg_lat
    overlap_deg_lon = (overlap_km * 1000) / m_per_deg_lon
    overlap_deg_lat = (overlap_km * 1000) / m_per_deg_lat

    step_lon_deg = tile_w_deg - overlap_deg_lon
    step_lat_deg = tile_h_deg - overlap_deg_lat

    r = 0
    lat_top = lat_max
    while lat_top > lat_min:
        lat_bottom = max(lat_top - tile_h_deg, lat_min)
        c = 0
        lon_left = lon_min
        while lon_left < lon_max:
            lon_right = min(lon_left + tile_w_deg, lon_max)
            yield (lon_left, lat_bottom, lon_right, lat_top), (r, c)
            c += 1
            lon_left = lon_left + step_lon_deg
        r += 1
        lat_top = lat_top - step_lat_deg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check presence of downloaded Sentinel-2 composite tiles for Brahmanbaria."
    )
    parser.add_argument("--month", default="october", help="Month (name or number). Default: october.")
    parser.add_argument("--year", type=int, default=2016, help="Year. Default: 2016.")
    parser.add_argument(
        "--tile-width-km",
        type=float,
        default=5.0,
        help="Tile width in kilometers. Default: 5.0",
    )
    parser.add_argument(
        "--tile-height-km",
        type=float,
        default=5.0,
        help="Tile height in kilometers. Default: 5.0",
    )
    parser.add_argument(
        "--tile-overlap-km",
        type=float,
        default=0.5,
        help="Tile overlap in kilometers. Default: 0.5",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/gee/tiles",
        help="Directory where tiles should be stored. Default: data/raw/gee/tiles",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix used when tiles were downloaded. Default derives from month/year, e.g., s2_oct_2016",
    )
    parser.add_argument(
        "--ext",
        default=".tif",
        help="File extension to check. Default: .tif",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    month_num = parse_month(args.month)
    month_abbr = calendar.month_abbr[month_num].lower()
    prefix = args.prefix or f"s2_{month_abbr}_{args.year}"
    bbox = bbox_from_coords(brahmanbaria_coords())
    tiles = list(
        split_bbox_by_km(bbox, args.tile_width_km, args.tile_height_km, args.tile_overlap_km)
    )

    out_dir = Path(args.output_dir)
    missing = []
    present = 0
    total = len(tiles)

    for _, (r, c) in tiles:
        fname = f"{prefix}_{args.year}_{month_num:02d}_r{r}_c{c}{args.ext}"
        fpath = out_dir / fname
        if fpath.exists():
            present += 1
        else:
            missing.append(fpath)

    print(f"Expected tiles: {total}")
    print(f"Present: {present}")
    print(f"Missing: {len(missing)}")
    if missing:
        print("Missing files:")
        for path in missing:
            print(f" - {path}")


if __name__ == "__main__":
    main()
