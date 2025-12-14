"""
Clip a raster (e.g., NDVI/EVI) in map/<year>/<metric>_<yyyymmdd>.tif to the Brahmanbaria AOI.
Output is saved to map/<year>/brahmanbaria_<metric>_<yyyymmdd>.tif with nodata outside AOI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import fiona
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from rasterio.mask import mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clip NDVI raster to Brahmanbaria AOI.")
    parser.add_argument(
        "--input",
        default="map/2017/ndvi_20171103.tif",
        help="Input raster path (pattern: map/<year>/<metric>_<yyyymmdd>.tif). Default: map/2017/ndvi_20171103.tif",
    )
    parser.add_argument(
        "--vector",
        default="map/brahmanbaria_gpkg.gpkg",
        help="AOI vector path. Default: map/brahmanbaria_gpkg.gpkg",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Layer name for multi-layer vectors. Default: first layer.",
    )
    return parser.parse_args()


def read_shapes(vector_path: Path, layer: str | None):
    with fiona.open(vector_path, layer=layer) as src:
        return [feature["geometry"] for feature in src], src.crs


def clip_raster(raster_path: Path, shapes, input_crs, output_path: Path) -> None:
    with rasterio.open(raster_path) as src:
        # Force output CRS to EPSG:32646
        target_crs = CRS.from_epsg(32646)
        if target_crs is None:
            raise SystemExit(f"Raster CRS is undefined for {raster_path}")
        # Reproject AOI shapes to raster CRS if needed
        if input_crs and input_crs != target_crs:
            shapes = [transform_geom(input_crs, target_crs, geom) for geom in shapes]

        fill_value = src.nodata if src.nodata is not None else np.nan
        out_image, out_transform = mask(src, shapes, crop=True, filled=True, nodata=fill_value)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": fill_value,
                "crs": target_crs,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)


def main() -> None:
    args = parse_args()
    ndvi_path = Path(args.input)
    if not ndvi_path.exists():
        raise SystemExit(f"Input raster not found: {ndvi_path}")

    year = ndvi_path.parent.name
    stem = ndvi_path.stem  # e.g., ndvi_20171103
    parts = stem.split("_")
    if len(parts) < 2:
        raise SystemExit(f"Cannot parse date/metric from filename: {stem}")
    metric = parts[0]
    date_str = parts[1]

    vector_path = Path(args.vector)
    if not vector_path.exists():
        raise SystemExit(f"Vector AOI not found: {vector_path}")

    shapes, input_crs = read_shapes(vector_path, args.layer)
    if not shapes:
        raise SystemExit(f"No geometries found in {vector_path}")

    out_path = Path("map") / year / f"brahmanbaria_{metric}_{date_str}.tif"
    clip_raster(ndvi_path, shapes, input_crs, out_path)
    print(f"Saved clipped NDVI to {out_path}")


if __name__ == "__main__":
    main()
