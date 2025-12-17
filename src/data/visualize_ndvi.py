"""
Visualize NDVI with categorical classes and save to an image file.

Defaults target map/<year>/brahmanbaria_ndvi_<yyyymmdd>.tif.
"""
from __future__ import annotations
import src.geo_env
import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize NDVI raster with categorical classes.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to NDVI GeoTIFF (e.g., map/2016/brahmanbaria_ndvi_20161113.tif). "
        "If omitted, defaults to map/<year>/brahmanbaria_ndvi_*.tif when --year is given.",
    )
    parser.add_argument("--year", type=int, default=2016, help="Year folder under map/ to search if --input not set.")
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(10, 10),
        help="Figure size (width height). Default: 10 10",
    )
    parser.add_argument(
        "--output-format",
        default="png",
        choices=["png", "jpg", "pdf"],
        help="Output format. Default: png",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=4,
        help="Downsample factor (integer > 0) to reduce memory usage; 1 means full res. Default: 4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        ndvi_path = args.input
    else:
        candidates = sorted((Path("map") / str(args.year)).glob("brahmanbaria_ndvi_*.tif"))
        if not candidates:
            raise SystemExit(f"No NDVI raster found in map/{args.year} matching brahmanbaria_ndvi_*.tif")
        if len(candidates) > 1:
            raise SystemExit(
                f"Multiple NDVI rasters found in map/{args.year}. Please pass one via --input:\n"
                + "\n".join(str(c) for c in candidates)
            )
        ndvi_path = candidates[0]

    if not ndvi_path.exists():
        raise SystemExit(f"NDVI raster not found: {ndvi_path}")

    # Extract date from filename (first 8-digit sequence)
    import re

    m = re.search(r"(\d{8})", ndvi_path.name)
    if not m:
        raise SystemExit("Could not find yyyymmdd date in input filename.")
    date_str = m.group(1)

    bounds = [-0.5, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.5]
    hex_colors = ["#08306B", "#D9D9D9", "#FEE391", "#A6D96A", "#66BD63", "#1A9850", "#006837"]
    labels = [
        "< 0 (Water/Snow/Clouds)",
        "0–0.1 (Bare/Urban)",
        "0.1–0.2 (Very Sparse)",
        "0.2–0.4 (Sparse–Moderate)",
        "0.4–0.6 (Moderate)",
        "0.6–0.8 (Dense)",
        "> 0.8 (Very Dense)",
    ]
    cmap = ListedColormap(hex_colors)
    norm = BoundaryNorm(bounds, cmap.N)
    ticks = [(-0.5 + 0) / 2, (0 + 0.1) / 2, (0.1 + 0.2) / 2, (0.2 + 0.4) / 2,
            (0.4 + 0.6) / 2, (0.6 + 0.8) / 2, (0.8 + 1.5) / 2]

    with rasterio.open(ndvi_path) as src:
        out_height = max(1, src.height // args.scale_factor)
        out_width = max(1, src.width // args.scale_factor)
        ndvi = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.average,
        )
        ndvi = np.ma.masked_invalid(ndvi)
        lon_min, lat_min, lon_max, lat_max = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    
    pad_frac = 0.03  # 3% padding outside frame
    pad_lon = (lon_max - lon_min) * pad_frac
    pad_lat = (lat_max - lat_min) * pad_frac
    ext_padded = [lon_min - pad_lon, lon_max +
                pad_lon, lat_min - pad_lat, lat_max + pad_lat]
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    true_extent = [lon_min, lon_max, lat_min, lat_max]
    im = ax.imshow(ndvi, cmap=cmap, norm=norm, extent=true_extent, origin="upper")
    ax.set_xlim(ext_padded[0], ext_padded[1])
    ax.set_ylim(ext_padded[2], ext_padded[3])
    ax.margins(0.02)
    
    ax.set_title(f"NDVI ({date_str[:4]}-{date_str[4:6]}-{date_str[6:]}) - Brahmanbaria", pad=12)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    fig.subplots_adjust(left=0.08, right=0.78, top=0.93, bottom=0.1, wspace=0.15)

    cax = fig.add_axes([0.82, 0.2, 0.04, 0.6])
    cbar = fig.colorbar(im, cax=cax, boundaries=bounds, ticks=ticks)
    cbar.ax.set_yticklabels(labels)
    cbar.ax.set_title("NDVI with Category", pad=6, fontsize=10)

    ax.annotate("N", xy=(0.97, 0.14), xytext=(0.97, 0.04),
                xycoords="axes fraction", ha="center", va="center",
                arrowprops=dict(facecolor="black", width=3, headwidth=10))

    scale_len_km = 10
    mid_lat = (lat_min + lat_max) / 2
    deg_per_km_lon = 1 / (111.32 * np.cos(np.deg2rad(mid_lat)))
    bar_len_lon = scale_len_km * deg_per_km_lon
    x0 = ext_padded[0] + 0.05 * (ext_padded[1] - ext_padded[0])
    y0 = ext_padded[2] + 0.05 * (ext_padded[3] - ext_padded[2])
    ax.plot([x0, x0 + bar_len_lon], [y0, y0], color="k", lw=3)
    ax.text(x0 + bar_len_lon / 2, y0 + (ext_padded[3] - ext_padded[2]) * 0.01,
            f"{scale_len_km} km", ha="center", va="bottom", fontsize=9)

    out_dir = Path("data/processed") / date_str[:4]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ndvi_brahmanbaria_{date_str}.{args.output_format}"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
