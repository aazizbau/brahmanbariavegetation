"""
Mosaic single-date Sentinel-2 tiles (e.g., from download_gee_s2.py) into a single GeoTIFF.

Assumes tiles follow the naming pattern:
<prefix>_<year>_<month>_<day>_r<row>_c<col>.tif
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import rasterio
from rasterio.windows import from_bounds


def find_tiles(input_dir: Path, prefix: str, year: int, month: int, day: int) -> List[Path]:
    pattern = f"{prefix}_{year}_{month:02d}_{day:02d}_r*_c*.tif"
    tiles = sorted(input_dir.glob(pattern))
    if not tiles:
        raise FileNotFoundError(f"No tiles found matching {pattern} in {input_dir}")
    return tiles


def chunked(iterable: List[Path], size: int) -> Iterable[List[Path]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def streaming_mosaic(tiles: List[Path], output_path: Path) -> None:
    if not tiles:
        raise ValueError("No tiles provided for mosaicking")

    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False, GDAL_PAM_ENABLED=False):
        with rasterio.open(tiles[0]) as ref:
            nodata = ref.nodata if ref.nodata is not None else 0
            dtype = ref.dtypes[0]
            count = ref.count
            transform = ref.transform
            crs = ref.crs
            res_x = transform.a
            res_y = -transform.e

        lefts, bottoms, rights, tops = [], [], [], []
        for t in tiles:
            with rasterio.open(t) as src:
                lefts.append(src.bounds.left)
                bottoms.append(src.bounds.bottom)
                rights.append(src.bounds.right)
                tops.append(src.bounds.top)
        min_left, min_bottom, max_right, max_top = min(lefts), min(bottoms), max(rights), max(tops)

        width = int(math.ceil((max_right - min_left) / res_x))
        height = int(math.ceil((max_top - min_bottom) / res_y))
        out_transform = rasterio.transform.from_origin(min_left, max_top, res_x, res_y)

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "transform": out_transform,
            "count": count,
            "dtype": dtype,
            "crs": crs,
            "photometric": "MINISBLACK",
            "nodata": nodata,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            for tile in tiles:
                with rasterio.open(tile) as src:
                    data = src.read()
                    win = from_bounds(
                        src.bounds.left,
                        src.bounds.bottom,
                        src.bounds.right,
                        src.bounds.top,
                        transform=out_transform,
                        height=height,
                        width=width,
                    )
                    dst.write(data, window=win)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic Sentinel-2 tiles into a single GeoTIFF.")
    parser.add_argument(
        "--input-dir",
        default="data/raw/gee/s2_single",
        help="Directory containing tile GeoTIFFs. Default: data/raw/gee/s2_single",
    )
    parser.add_argument(
        "--prefix",
        default="s2_20171103",
        help="Tile filename prefix (e.g., s2_20171103). Default: s2_20171103",
    )
    parser.add_argument("--year", type=int, default=2017, help="Year used in filenames. Default: 2017")
    parser.add_argument("--month", type=int, default=11, help="Month used in filenames. Default: 11")
    parser.add_argument("--day", type=int, default=3, help="Day used in filenames. Default: 3")
    parser.add_argument(
        "--output",
        default="data/interim/mosaic_s2_20171103.tif",
        help="Output mosaicked GeoTIFF path. Default: data/interim/mosaic_s2_20171103.tif",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    tiles = find_tiles(input_dir, args.prefix, args.year, args.month, args.day)
    print(f"Found {len(tiles)} tiles. Mosaicking...")
    streaming_mosaic(tiles, output_path)
    print(f"Saved mosaic to {output_path}")


if __name__ == "__main__":
    main()
