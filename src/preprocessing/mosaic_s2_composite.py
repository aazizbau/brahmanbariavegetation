"""
Mosaic Sentinel-2 composite tiles into a single GeoTIFF.

Assumes tiles follow the naming pattern used by the download scripts:
`<prefix>_<year>_<month>_r<row>_c<col>.tif`, all in the same folder and with
matching projection and resolution.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import rasterio
from rasterio.windows import from_bounds


def find_tiles(input_dir: Path, prefix: str, year: int, month: int) -> List[Path]:
    pattern = f"{prefix}_{year}_{month:02d}_r*_c*.tif"
    tiles = sorted(input_dir.glob(pattern))
    if not tiles:
        raise FileNotFoundError(f"No tiles found matching {pattern} in {input_dir}")
    return tiles


def _streaming_mosaic(tiles: List[Path], output_path: Path) -> Tuple[rasterio.Affine, dict]:
    """
    Write a mosaic by streaming tiles into the output file without holding the full
    mosaic in memory. Last-write-wins for overlaps.
    """
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

        # Compute overall bounds
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

    return out_transform, profile


def chunked(iterable: List[Path], size: int) -> Iterable[List[Path]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def mosaic_tiles(tiles: List[Path], output_path: Path, batch_size: int = 100, temp_dir: Path | None = None):
    """
    Mosaic tiles, optionally in batches to reduce memory usage.
    """
    if len(tiles) <= batch_size:
        return _streaming_mosaic(tiles, output_path)

    temp_dir = temp_dir or output_path.parent / "tmp_mosaic"
    temp_dir.mkdir(parents=True, exist_ok=True)
    batch_outputs = []
    for idx, chunk in enumerate(chunked(tiles, batch_size), start=1):
        batch_out = temp_dir / f"batch_{idx:03d}.tif"
        print(f"Mosaicking batch {idx} with {len(chunk)} tiles -> {batch_out}")
        _streaming_mosaic(chunk, batch_out)
        batch_outputs.append(batch_out)

    print(f"Mosaicking {len(batch_outputs)} batch outputs into final mosaic...")
    return _streaming_mosaic(batch_outputs, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic Sentinel-2 composite tiles into a single GeoTIFF.")
    parser.add_argument(
        "--input-dir",
        default="data/raw/gee/tiles",
        help="Directory containing tile GeoTIFFs. Default: data/raw/gee/tiles",
    )
    parser.add_argument(
        "--prefix",
        default="s2_oct_2016",
        help="Tile filename prefix (e.g., s2_oct_2016). Default: s2_oct_2016",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of tiles per batch during mosaic (helps with memory). Default: 100",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Temporary directory for batch mosaics. Default: <output_dir>/tmp_mosaic",
    )
    parser.add_argument("--year", type=int, default=2016, help="Year used in filenames. Default: 2016")
    parser.add_argument("--month", type=int, default=10, help="Month used in filenames (01-12). Default: 10")
    parser.add_argument(
        "--output",
        default="data/interim/mosaic_s2.tif",
        help="Output mosaicked GeoTIFF path. Default: data/interim/mosaic_s2.tif",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tiles = find_tiles(input_dir, args.prefix, args.year, args.month)
    print(f"Found {len(tiles)} tiles. Mosaicking...")
    _, meta = mosaic_tiles(tiles, output_path, batch_size=args.batch_size, temp_dir=Path(args.temp_dir) if args.temp_dir else None)
    print(f"Saved mosaic to {output_path} with CRS {meta.get('crs')} and shape {meta.get('height')}x{meta.get('width')}")


if __name__ == "__main__":
    main()
