"""
Download a single-date cloud-masked Sentinel-2 L2A image (not a composite)
for the Brahmanbaria AOI, split into tiles to avoid EE download size limits.

Cloud masking uses QA60 bits 10/11 (cloud/cirrus) and scales reflectance by 1/10000.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

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


def mask_s2_clouds(image):
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)


def meters_per_degree(mean_lat: float) -> tuple[float, float]:
    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * math.cos(math.radians(mean_lat))
    return meters_per_deg_lat, meters_per_deg_lon


def bbox_from_coords(coords: list[list[float]]) -> tuple[float, float, float, float]:
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


def bbox_coords(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> list:
    return [
        [lon_min, lat_max],
        [lon_max, lat_max],
        [lon_max, lat_min],
        [lon_min, lat_min],
        [lon_min, lat_max],
    ]


def split_bbox_by_km(
    bbox: tuple[float, float, float, float], tile_w_km: float, tile_h_km: float, overlap_km: float
) -> list[tuple[tuple[float, float, float, float], tuple[int, int]]]:
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

    tiles = []
    r = 0
    lat_top = lat_max
    while lat_top > lat_min:
        lat_bottom = max(lat_top - tile_h_deg, lat_min)
        c = 0
        lon_left = lon_min
        while lon_left < lon_max:
            lon_right = min(lon_left + tile_w_deg, lon_max)
            tiles.append(((lon_left, lat_bottom, lon_right, lat_top), (r, c)))
            c += 1
            lon_left = lon_left + step_lon_deg
        r += 1
        lat_top = lat_top - step_lat_deg
    return tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download single-date cloud-masked Sentinel-2 L2A for Brahmanbaria AOI with tiling."
    )
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format (e.g., 2017-03-11).")
    parser.add_argument(
        "--output-dir",
        default="data/raw/gee/s2_single",
        help="Directory to store tile GeoTIFFs. Default: data/raw/gee/s2_single",
    )
    parser.add_argument("--scale", type=float, default=10, help="Scale (meters). Default: 10.")
    parser.add_argument("--crs", default="EPSG:4326", help="Output CRS. Default: EPSG:4326.")
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
        "--prefix",
        default=None,
        help="Filename prefix for tiles. Default derives from date, e.g., s2_20171103",
    )
    return parser.parse_args()


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
    print(f"[{idx}/{total}] Saved {out_path}")


def main() -> None:
    args = parse_args()
    date_str = args.date
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    next_date = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    prefix = args.prefix or f"s2_{dt.strftime('%Y%m%d')}"

    ee.Initialize()
    aoi = ee.Geometry.Polygon([brahmanbaria_coords()])
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(date_str, next_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(mask_s2_clouds)
    )

    if collection.size().getInfo() == 0:
        raise SystemExit(f"No images found on {date_str} for the AOI.")

    # If multiple tiles exist that day, take the median to merge them
    image = collection.median().select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"])

    bbox = bbox_from_coords(brahmanbaria_coords())
    tiles = split_bbox_by_km(bbox, args.tile_width_km, args.tile_height_km, args.tile_overlap_km)
    out_dir = Path(args.output_dir)

    print(
        f"Downloading {len(tiles)} tiles for {date_str} at {args.scale} m "
        f"(tile {args.tile_width_km}x{args.tile_height_km} km, overlap {args.tile_overlap_km} km)..."
    )
    total = len(tiles)
    for idx, ((lon_min, lat_min, lon_max, lat_max), (r, c)) in enumerate(tiles, start=1):
        tile_geom = ee.Geometry.Polygon([bbox_coords(lon_min, lat_min, lon_max, lat_max)])
        out_path = out_dir / f"{prefix}_{dt.year}_{dt.month:02d}_{dt.day:02d}_r{r}_c{c}.tif"
        download_tile(image, tile_geom, args.scale, args.crs, out_path, idx, total)


if __name__ == "__main__":
    main()
