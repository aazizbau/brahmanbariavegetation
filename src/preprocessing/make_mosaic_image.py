"""
Mosaic Sentinel-2 band tiles (e.g., B02) from local SAFE folders using SCL masks.

- Searches under F:/brahmanbaria_images/<year>/ for SAFE granules.
- Reads the requested band at 10 m (R10m) and the corresponding SCL at 20 m (R20m),
  masks invalid SCL classes (0,1,3,8,9,10,11) and zeros, forces CRS to EPSG:32646,
  and mosaics all tiles.
- Outputs to map/<year>/S2_<yyyymmdd>_<band>_10m.tif by default.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import tempfile
import math
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.windows import from_bounds

OUT_CRS = CRS.from_epsg(32646)
INVALID_SCL = {0, 1, 3, 8, 9, 10, 11}
BASE_DIR = Path("F:/brahmanbaria_images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic Sentinel-2 band tiles with SCL masking.")
    parser.add_argument("--year", type=int, default=2017, help="Year folder under F:/brahmanbaria_images/. Default: 2017")
    parser.add_argument("--band", default="B02", help="Band code (e.g., B02, B03, B04, B08). Default: B02")
    parser.add_argument(
        "--output",
        default=None,
        help="Output GeoTIFF path. Default: map/<year>/S2_<date>_<band>_10m.tif (date inferred from filenames).",
    )
    return parser.parse_args()


def find_band_tiles(year: int, band: str) -> List[tuple[Path, Path]]:
    base = BASE_DIR / str(year)
    pattern = f"**/GRANULE/*/IMG_DATA/R10m/*_{band}_10m.jp2"
    band_files = sorted(base.glob(pattern))
    tiles = []
    for bpath in band_files:
        # infer granule root to find SCL
        granule_dir = bpath.parent.parent  # IMG_DATA
        scl_candidates = list(granule_dir.joinpath("R20m").glob("*_SCL_20m.jp2"))
        if not scl_candidates:
            continue
        tiles.append((bpath, scl_candidates[0]))
    return tiles


def extract_date_from_name(path: Path) -> Optional[str]:
    # Expect substring like 20171103 in filename
    m = re.search(r"20\d{6}", path.name)
    if m:
        return m.group(0)
    return None


def main() -> None:
    args = parse_args()
    tiles = find_band_tiles(args.year, args.band)
    if not tiles:
        raise SystemExit(f"No tiles found for band {args.band} in {BASE_DIR / str(args.year)}")

    # Determine date from first band filename
    date_str = extract_date_from_name(tiles[0][0]) or "unknown"
    out_path = Path(args.output) if args.output else Path("map") / str(args.year) / f"S2_{date_str}_{args.band}_10m.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    masked_datasets = []
    temp_files = []
    try:
        for bpath, scl_path in tiles:
            with rasterio.open(bpath) as b_ds, rasterio.open(scl_path) as scl_ds:
                b_arr = b_ds.read(1).astype(np.float32)
                scl = scl_ds.read(
                    1,
                    out_shape=b_arr.shape,
                    resampling=Resampling.nearest,
                )
                # Build boolean mask without large temporary arrays
                mask = np.zeros_like(b_arr, dtype=bool)
                for val in INVALID_SCL:
                    mask |= scl == val
                mask |= b_arr == 0
                b_arr[mask] = np.nan

                profile = b_ds.profile.copy()
                profile.update(
                    driver="GTiff",
                    dtype="float32",
                    nodata=np.nan,
                    crs=OUT_CRS,
                    compress="deflate",
                    tiled=True,
                )
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".tif", delete=False, dir=out_path.parent
                )
                tmp_path = Path(tmp.name)
                tmp.close()
                with rasterio.open(tmp_path, "w", **profile) as tmp_ds:
                    tmp_ds.write(b_arr, 1)
                temp_files.append(tmp_path)
                masked_datasets.append(rasterio.open(tmp_path))

        # Streaming mosaic to reduce memory
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False, GDAL_PAM_ENABLED=False):
            with rasterio.open(masked_datasets[0].name) as ref:
                nodata = ref.nodata
                dtype = ref.dtypes[0]
                count = ref.count
                transform = ref.transform
                crs = OUT_CRS
                res_x = transform.a
                res_y = -transform.e

            lefts, bottoms, rights, tops = [], [], [], []
            for t in masked_datasets:
                with rasterio.open(t.name) as src:
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
                "compress": "deflate",
                "tiled": True,
            }

            with rasterio.open(out_path, "w", **profile) as dst:
                for t in masked_datasets:
                    with rasterio.open(t.name) as src:
                        win = from_bounds(
                            src.bounds.left,
                            src.bounds.bottom,
                            src.bounds.right,
                            src.bounds.top,
                            transform=out_transform,
                            height=height,
                            width=width,
                        )
                        for _, src_window in src.block_windows(1):
                            dst_window = rasterio.windows.Window(
                                col_off=int(win.col_off + src_window.col_off),
                                row_off=int(win.row_off + src_window.row_off),
                                width=src_window.width,
                                height=src_window.height,
                            )
                            data = src.read(1, window=src_window, masked=True).filled(np.nan)
                            dst.write(data, 1, window=dst_window)

        print(f"✅ Mosaic written successfully\n📌 CRS: {OUT_CRS}\n📁 Output: {out_path}")
    finally:
        for ds in masked_datasets:
            ds.close()
        for t in temp_files:
            try:
                t.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
