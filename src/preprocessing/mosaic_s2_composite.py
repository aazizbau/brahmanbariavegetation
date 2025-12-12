"""
Mosaic Sentinel-2 composite tiles into a single GeoTIFF.

Assumes tiles follow the naming pattern used by the download scripts:
`<prefix>_<year>_<month>_r<row>_c<col>.tif`, all in the same folder and with
matching projection and resolution.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import List, Tuple

import rasterio
from rasterio.merge import merge


def find_tiles(input_dir: Path, prefix: str, year: int, month: int) -> List[Path]:
    pattern = f"{prefix}_{year}_{month:02d}_r*_c*.tif"
    tiles = sorted(input_dir.glob(pattern))
    if not tiles:
        raise FileNotFoundError(f"No tiles found matching {pattern} in {input_dir}")
    return tiles


def mosaic_tiles(tiles: List[Path], output_path: Path) -> Tuple[rasterio.Affine, dict]:
    with ExitStack() as stack:
        # Disable internal masks/PAM to reduce TIFF metadata warnings from GEE exports
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False, GDAL_PAM_ENABLED=False):
            src_files_to_mosaic = [stack.enter_context(rasterio.open(tile)) for tile in tiles]
            mosaic, out_trans = merge(src_files_to_mosaic)
            out_meta = src_files_to_mosaic[0].meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": mosaic.shape[0],
            "photometric": "MINISBLACK",  # avoid color/extra-samples mismatch warnings for multispectral rasters
        }
    )
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    return out_trans, out_meta


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
    _, meta = mosaic_tiles(tiles, output_path)
    print(f"Saved mosaic to {output_path} with CRS {meta.get('crs')} and shape {meta.get('height')}x{meta.get('width')}")


if __name__ == "__main__":
    main()
