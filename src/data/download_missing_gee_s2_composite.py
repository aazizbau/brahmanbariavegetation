"""
Download only missing Sentinel-2 composite tiles for the Brahmanbaria AOI.
Uses the same tiling scheme as download_gee_s2_composite.py (tile size in km
with optional overlap) and the same default filename pattern
`s2_<monthabbr>_<year>_<year>_<month>_r{row}_c{col}.tif`.
"""

from __future__ import annotations

import argparse
import calendar
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import ee
import requests


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


def init_ee() -> None:
    service_account = os.getenv("EE_SERVICE_ACCOUNT")
    key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if service_account and key_file and Path(key_file).exists():
        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials)
    else:
        ee.Initialize()


def build_composite(
    aoi: ee.Geometry, start: str, end: str, scale: float, crs: str, collection_key: str
) -> ee.Image:
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]
    collection_id = {
        "sr_harmonized": "COPERNICUS/S2_SR_HARMONIZED",
        "l1c": "COPERNICUS/S2_HARMONIZED",
    }.get(collection_key)
    if not collection_id:
        raise ValueError(f"Unsupported collection: {collection_key}")

    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    )

    if collection.size().getInfo() == 0:
        raise SystemExit(
            f"No images found for {start} to {end} in collection {collection_id}. "
            "Try a different month/year or relax filters."
        )

    return collection.median().select(bands).resample("bilinear").reproject(crs=crs, scale=scale)


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


def bbox_coords(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> list:
    return [
        [lon_min, lat_max],
        [lon_max, lat_max],
        [lon_max, lat_min],
        [lon_min, lat_min],
        [lon_min, lat_max],
    ]


def download_tile(
    image: ee.Image,
    tile_geom: ee.Geometry,
    scale: float,
    crs: str,
    out_path: Path,
    idx: int,
    total: int,
) -> None:
    region = tile_geom.coordinates().getInfo()
    url = image.clip(tile_geom).getDownloadURL(
        {"region": region, "scale": scale, "crs": crs, "format": "GEO_TIFF"}
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {idx}/{total} downloaded, Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download missing Sentinel-2 composite tiles for Brahmanbaria from Earth Engine."
    )
    parser.add_argument("--month", default="october", help="Month (name or number). Default: october.")
    parser.add_argument("--year", type=int, default=2016, help="Year. Default: 2016.")
    parser.add_argument(
        "--scale",
        type=float,
        default=10,
        help="Target resolution in meters. Default: 10 (use 20 if hitting size limits).",
    )
    parser.add_argument("--crs", default="EPSG:4326", help="Output CRS. Default: EPSG:4326.")
    parser.add_argument(
        "--collection",
        choices=["sr_harmonized", "l1c"],
        default="sr_harmonized",
        help="Sentinel-2 collection: sr_harmonized (L2A) or l1c (TOA). Default: sr_harmonized.",
    )
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
        help="Output directory for tiled GeoTIFFs. Default: data/raw/gee/tiles",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix for tiles. Default derives from month/year, e.g., s2_oct_2016",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    month_num = parse_month(args.month)
    month_abbr = calendar.month_abbr[month_num].lower()
    prefix = args.prefix or f"s2_{month_abbr}_{args.year}"

    # Determine expected tiles
    bbox = bbox_from_coords(brahmanbaria_coords())
    tiles = list(
        split_bbox_by_km(bbox, args.tile_width_km, args.tile_height_km, args.tile_overlap_km)
    )

    out_dir = Path(args.output_dir)
    missing = []
    for bbox, (r, c) in tiles:
        fname = f"{prefix}_{args.year}_{month_num:02d}_r{r}_c{c}.tif"
        fpath = out_dir / fname
        if not fpath.exists():
            missing.append((bbox, fpath, r, c))

    if not missing:
        print("All tiles present. Nothing to download.")
        return

    # Build composite once
    start = f"{args.year}-{month_num:02d}-01"
    _, last_day = calendar.monthrange(args.year, month_num)
    end = f"{args.year}-{month_num:02d}-{last_day} 23:59:59"

    init_ee()
    aoi = ee.Geometry.Polygon([brahmanbaria_coords()])
    composite = build_composite(aoi, start, end, args.scale, args.crs, args.collection)

    total = len(missing)
    print(f"{total} tiles missing; downloading to {out_dir} ...")
    for idx, (tile_bbox, fpath, r, c) in enumerate(missing, start=1):
        lon_min, lat_min, lon_max, lat_max = tile_bbox
        tile_geom = ee.Geometry.Polygon([bbox_coords(lon_min, lat_min, lon_max, lat_max)])
        download_tile(composite, tile_geom, args.scale, args.crs, fpath, idx, total)


if __name__ == "__main__":
    main()
