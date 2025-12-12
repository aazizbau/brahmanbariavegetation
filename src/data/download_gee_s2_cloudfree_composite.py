"""
Download a cloud-filtered monthly median Sentinel-2 composite from Google Earth
Engine for the Brahmanbaria AOI, using S2 cloud probability and SCL masks.

Masking rules:
- Cloud probability < threshold (default 40).
- SCL classes kept: 4 (vegetation), 5 (bare soil), 6 (water).

Tiles are downloaded locally (splitting the AOI into km-sized tiles to avoid
EE download limits). Bands: B2, B3, B4, B5, B6, B7, B8, B8A.
"""

from __future__ import annotations

import argparse
import calendar
import math
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


def month_range(year: int, month: int) -> Tuple[str, str]:
    start = datetime(year, month, 1)
    _, last_day = calendar.monthrange(year, month)
    end = datetime(year, month, last_day, 23, 59, 59)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def meters_per_degree(mean_lat: float) -> Tuple[float, float]:
    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * math.cos(math.radians(mean_lat))
    return meters_per_deg_lat, meters_per_deg_lon


def bbox_from_coords(coords: List[List[float]]) -> Tuple[float, float, float, float]:
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


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


def init_ee() -> None:
    ee.Initialize()


def build_cloud_masked_collection(aoi: ee.Geometry, start: str, end: str, cp_thresh: int) -> ee.ImageCollection:
    s2_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
    )

    if s2_sr.size().getInfo() > 0:
        cp = (
            ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterBounds(aoi)
            .filterDate(start, end)
        )

        def add_mask_scl_only(img):
            scl = img.select("SCL")
            keep_scl = (
                scl.eq(4)
                .Or(scl.eq(5))
                .Or(scl.eq(6))
                .Or(scl.eq(7))
                .Or(scl.eq(11))
            )
            return img.updateMask(keep_scl)

        # If no cloud probability scenes exist (common pre-2017), fall back to SCL-only mask.
        if cp.size().getInfo() == 0:
            return s2_sr.map(add_mask_scl_only)

        join = ee.Join.saveFirst("cloud_mask")
        joined = join.apply(
            primary=s2_sr,
            secondary=cp,
            condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
        )

        def add_mask(img):
            cloud_prob = ee.Image(img.get("cloud_mask")).select("probability")
            scl = img.select("SCL")
            cloud_free = cloud_prob.lt(cp_thresh)
            keep_scl = (
                scl.eq(4)
                .Or(scl.eq(5))
                .Or(scl.eq(6))
                .Or(scl.eq(7))
                .Or(scl.eq(11))
            )
            mask = cloud_free.And(keep_scl)
            return img.updateMask(mask)

        return ee.ImageCollection(joined).map(add_mask)

    # Fallback to L1C with QA60-based masking when SR is unavailable (e.g., early 2016).
    s2_l1c = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
    )

    def add_mask_qa60(img):
        qa = img.select("QA60")
        cloud = qa.bitwiseAnd(1 << 10).neq(0)
        cirrus = qa.bitwiseAnd(1 << 11).neq(0)
        mask = cloud.Not().And(cirrus.Not())
        return img.updateMask(mask)

    return s2_l1c.map(add_mask_qa60)


def build_composite(
    aoi: ee.Geometry, start: str, end: str, scale: float, crs: str, cp_thresh: int
) -> ee.Image:
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]
    collection = build_cloud_masked_collection(aoi, start, end, cp_thresh)

    if collection.size().getInfo() == 0:
        raise SystemExit(
            f"No images found for {start} to {end} after filtering. "
            "Try a different month/year or relax filters."
        )

    composite = collection.median().select(bands).resample("bilinear").reproject(crs=crs, scale=scale)
    return composite


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
        description="Download cloud-masked monthly median Sentinel-2 composite for Brahmanbaria from Earth Engine."
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
        "--cloud-prob-threshold",
        type=int,
        default=40,
        help="Cloud probability threshold (0-100). Pixels below this are kept. Default: 40.",
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
        help="Filename prefix for tiles. Default derives from month/year, e.g., s2_oct_2016_cf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    month_num = parse_month(args.month)
    start, end = month_range(args.year, month_num)
    month_abbr = calendar.month_abbr[month_num].lower()
    prefix = args.prefix or f"s2_{month_abbr}_{args.year}_cf"

    bbox = bbox_from_coords(brahmanbaria_coords())
    tiles = list(
        split_bbox_by_km(bbox, args.tile_width_km, args.tile_height_km, args.tile_overlap_km)
    )
    n_bands = 8
    out_dir = Path(args.output_dir)

    init_ee()
    aoi = ee.Geometry.Polygon([brahmanbaria_coords()])
    composite = build_composite(aoi, start, end, args.scale, args.crs, args.cloud_prob_threshold)

    total_tiles = len(tiles)
    print(
        f"Downloading {total_tiles} cloud-masked tiles at {args.scale} m for {args.year}-{month_num:02d} "
        f"(tile {args.tile_width_km}x{args.tile_height_km} km, overlap {args.tile_overlap_km} km)..."
    )
    for idx, ((lon_min, lat_min, lon_max, lat_max), (r, c)) in enumerate(tiles, start=1):
        tile_geom = ee.Geometry.Polygon([bbox_coords(lon_min, lat_min, lon_max, lat_max)])
        out_path = out_dir / f"{prefix}_{args.year}_{month_num:02d}_r{r}_c{c}.tif"
        download_tile(composite, tile_geom, args.scale, args.crs, out_path, idx, total_tiles)


if __name__ == "__main__":
    main()
