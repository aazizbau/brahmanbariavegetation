"""
Clip Brahmanbaria natural forest mosaic to the AOI boundary.

Usage:
    python src/naturalforest/clip_naturalforest.py \
        --input map/2020/naturalforest/brahmanbaria_naturalforest_2020_mosaic.tif \
        --output map/2020/naturalforest/brahmanbaria_naturalforest_2020_clip.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Ensure project root is on sys.path so geo_env can set PROJ paths.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.geo_env  # noqa: F401  # MUST come immediately after path fix

import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom


DEFAULT_VECTOR = Path("map/brahmanbaria_gpkg.gpkg")


def read_shapes(vector_path: Path) -> Sequence[dict]:
    """Load AOI shapes from a vector file."""
    with fiona.open(vector_path) as src:
        return [feature["geometry"] for feature in src]


def clip_raster(input_path: Path, shapes: Sequence[dict], output_path: Path) -> None:
    """Clip raster to AOI shapes and write output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError("Input raster CRS is undefined.")

        transformed_shapes = [
            transform_geom("EPSG:4326", src_crs, shape, precision=6)
            for shape in shapes
        ]

        out_image, out_transform = mask(
            src,
            transformed_shapes,
            crop=True,
            filled=True,
            nodata=src.nodata,
        )

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clip natural forest mosaic to AOI.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("map/2020/naturalforest/brahmanbaria_naturalforest_2020_mosaic.tif"),
        help="Input mosaic GeoTIFF.",
    )
    parser.add_argument(
        "--vector",
        type=Path,
        default=DEFAULT_VECTOR,
        help="Vector AOI (default: map/brahmanbaria_gpkg.gpkg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("map/2020/naturalforest/brahmanbaria_naturalforest_2020_clip.tif"),
        help="Output clipped GeoTIFF.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    shapes = read_shapes(args.vector)
    if not shapes:
        raise ValueError(f"No features found in {args.vector}")
    clip_raster(args.input, shapes, args.output)
    print(f"Saved clipped raster to {args.output}")


if __name__ == "__main__":
    main()
