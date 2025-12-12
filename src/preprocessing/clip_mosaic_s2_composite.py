"""
Clip a mosaicked Sentinel-2 raster to the Brahmanbaria AOI using a vector boundary.

Defaults:
- Input raster: data/interim/mosaic_s2_oct_2016.tif
- Vector AOI: map/brahmanbaria_gpkg.gpkg (first layer)
- Output raster: data/processed/mosaic_s2_oct_2016_clipped.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fiona
import rasterio
from rasterio.mask import mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clip mosaicked Sentinel-2 composite to AOI boundary.")
    parser.add_argument(
        "--input",
        default="data/interim/mosaic_s2_oct_2016.tif",
        help="Input mosaic raster path. Default: data/interim/mosaic_s2_oct_2016.tif",
    )
    parser.add_argument(
        "--vector",
        default="map/brahmanbaria_gpkg.gpkg",
        help="Vector AOI path (GeoPackage/GeoJSON/Shapefile). Default: map/brahmanbaria_gpkg.gpkg",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Layer name in vector (for multi-layer GPKG). Default: first layer.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/mosaic_s2_oct_2016_clipped.tif",
        help="Output clipped raster path. Default: data/processed/mosaic_s2_oct_2016_clipped.tif",
    )
    return parser.parse_args()


def read_shapes(vector_path: Path, layer: str | None):
    with fiona.open(vector_path, layer=layer) as src:
        return [feature["geometry"] for feature in src]


def clip_raster(raster_path: Path, shapes, output_path: Path) -> None:
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)


def main() -> None:
    args = parse_args()
    raster_path = Path(args.input)
    vector_path = Path(args.vector)
    output_path = Path(args.output)

    if not raster_path.exists():
        raise SystemExit(f"Input raster not found: {raster_path}")
    if not vector_path.exists():
        raise SystemExit(f"Vector AOI not found: {vector_path}")

    shapes = read_shapes(vector_path, args.layer)
    if not shapes:
        raise SystemExit(f"No geometries found in {vector_path}")

    clip_raster(raster_path, shapes, output_path)
    print(f"Saved clipped raster to {output_path}")


if __name__ == "__main__":
    main()
