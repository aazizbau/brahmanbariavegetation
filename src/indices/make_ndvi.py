"""
Compute NDVI from a mosaicked/clipped Sentinel-2 raster and write to a GeoTIFF.

Assumes the input raster band order follows the download pipeline:
B2, B3, B4, B5, B6, B7, B8, B8A (Band 4 = red, Band 8 = NIR).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute NDVI from a Sentinel-2 mosaic.")
    parser.add_argument(
        "--input",
        default="data/processed/mosaic_s2_oct_2016_clipped.tif",
        help="Input mosaic raster with bands B2,B3,B4,B5,B6,B7,B8,B8A. Default: data/processed/mosaic_s2_oct_2016_clipped.tif",
    )
    parser.add_argument(
        "--output",
        default="data/processed/ndvi_mosaic_s2_oct_2016.tif",
        help="Output NDVI GeoTIFF path. Default: data/processed/ndvi_mosaic_s2_oct_2016.tif",
    )
    parser.add_argument(
        "--red-band-index",
        type=int,
        default=None,
        help="1-based band index for red. Default: inferred from descriptions or 3 (B4).",
    )
    parser.add_argument(
        "--nir-band-index",
        type=int,
        default=None,
        help="1-based band index for NIR. Default: inferred from descriptions or 7 (B8).",
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


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Input raster not found: {in_path}")

    with rasterio.open(in_path) as src:
        # Band order from download pipeline: 1:B2, 2:B3, 3:B4, 4:B5, 5:B6, 6:B7, 7:B8, 8:B8A
        # Prefer band descriptions if present to avoid indexing mistakes; allow override via args.
        descriptions = [d or "" for d in src.descriptions]
        if args.red_band_index and args.nir_band_index:
            red_idx, nir_idx = args.red_band_index, args.nir_band_index
        else:
            try:
                red_idx = descriptions.index("B4") + 1
                nir_idx = descriptions.index("B8") + 1
            except ValueError:
                red_idx, nir_idx = 3, 7

        red = src.read(red_idx)
        nir = src.read(nir_idx)
        nodata = src.nodata
        nodata_mask = red == nodata if nodata is not None else None
        ndvi = compute_ndvi(red, nir, nodata_mask)

        profile = src.profile.copy()
        profile.update(
            {
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
            }
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ndvi.astype("float32"), 1)

    # Quick stats for sanity
    valid = ndvi[np.isfinite(ndvi)]
    print(
        f"Saved NDVI to {out_path} | red band {red_idx} nir band {nir_idx} | "
        f"ndvi min {valid.min() if valid.size else 'nan'} max {valid.max() if valid.size else 'nan'}"
    )


if __name__ == "__main__":
    main()
