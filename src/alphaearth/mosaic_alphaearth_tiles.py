"""
Fast mosaic for AlphaEarth tiles using GDAL VRT + single gdalwarp (Brahmanbaria AOI).

Example:
    python src/alphaearth/mosaic_alphaearth_tiles.py \
        --year 2017 \
        --input-base map/2017/alphaearth/brahmanbaria_alphaearth_2017.tif \
        --output map/2017/alphaearth/brahmanbaria_alphaearth_2017_mosaic.tif

Notes:
- Requires GDAL command-line tools on PATH: gdalbuildvrt, gdalwarp.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

import rasterio

# ---------------------------------------------------------------------------
# Optional proj.db helper (mirrors notebook behavior; safe if missing)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
geo_env_path = PROJECT_ROOT / "src" / "geo_env.py"
if geo_env_path.exists():
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    try:  # pragma: no cover
        import src.geo_env  # noqa: F401
    except Exception:
        pass

# Allow running as script or module
try:
    from . import download_alphaearth_embeddings as downloader
except Exception:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    import download_alphaearth_embeddings as downloader  # type: ignore


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def require_cmd(cmd: str) -> str:
    """Return resolved command path or raise a helpful error."""
    resolved = shutil.which(cmd)
    if not resolved:
        raise RuntimeError(
            f"Required command '{cmd}' not found on PATH. "
            "Install GDAL and ensure gdalbuildvrt/gdalwarp are available."
        )
    return resolved


def list_tile_paths(base: Path) -> list[Path]:
    """Return sorted tile paths matching the downloader naming scheme."""
    parent, stem, suffix = downloader.resolve_output_template(base)
    pattern = f"{stem}_r*_c*{suffix}"
    paths = sorted(parent.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No tiles found matching pattern {pattern} in directory {parent}"
        )
    return paths


def read_reference(tile_paths: list[Path]) -> dict:
    """Read reference metadata from the first tile (CRS, res, dtype, nodata)."""
    with rasterio.open(tile_paths[0]) as ds:
        meta = ds.meta.copy()
        nodata = ds.nodata
    if nodata is None:
        nodata = 0
    meta["nodata"] = nodata
    return meta


def run(cmd: list[str], *, env: dict[str, str] | None = None, allow_fail: bool = False) -> bool:
    """Run a subprocess; return success flag. Raise if allow_fail is False."""
    log(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        if allow_fail:
            log(f"Command failed (continuing due to allow_fail): {e}")
            return False
        raise RuntimeError(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}") from e


def build_vrt(tile_paths: list[Path], vrt_path: Path, *, ref_crs_wkt: str | None) -> None:
    """Build a VRT from tile paths, using an input list to avoid long command lines."""
    gdalbuildvrt = require_cmd("gdalbuildvrt")

    # Write the tile list to a temp file to avoid Windows command-length issues.
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        for p in tile_paths:
            f.write(str(p) + "\n")
        list_path = Path(f.name)

    cmd = [
        gdalbuildvrt,
        "-overwrite",
        "-resolution",
        "highest",
        "-input_file_list",
        str(list_path),
    ]
    if ref_crs_wkt:
        cmd += ["-a_srs", ref_crs_wkt]
    cmd += [str(vrt_path)]
    try:
        run(cmd)
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass


def warp_to_geotiff(
    vrt_path: Path,
    output: Path,
    *,
    nodata: float | int,
    dtype: str,
    gdal_cache_mb: int,
    compress: str,
    tile_size: int,
    overwrite: bool,
) -> None:
    """Convert VRT to GeoTIFF with gdalwarp in one pass."""
    gdalwarp = require_cmd("gdalwarp")
    env = os.environ.copy()
    env["GDAL_CACHEMAX"] = str(gdal_cache_mb)
    env["GDAL_NUM_THREADS"] = "2"

    creation_opts = [
        "-co",
        "TILED=YES",
        "-co",
        f"BLOCKXSIZE={tile_size}",
        "-co",
        f"BLOCKYSIZE={tile_size}",
        "-co",
        "BIGTIFF=YES",
    ]
    if compress.upper() != "NONE":
        creation_opts += ["-co", f"COMPRESS={compress}"]
    cmd = [
        gdalwarp,
        # REMOVE -multi
        "-wo", "NUM_THREADS=2",
        "-wm", "256",   # <-- critical,
        "-srcnodata",
        str(nodata),
        "-dstnodata",
        str(nodata),
        "-ot",
        dtype,
    ]
    if overwrite:
        cmd.append("-overwrite")
    cmd += creation_opts
    cmd += [str(vrt_path), str(output)]
    run(cmd, env=env)


def clean_tiles(tile_paths: list[Path], temp_dir: Path, skip_errors: bool) -> list[Path]:
    """
    Re-encode tiles to float32 GTiffs without compression/predictor.
    All bands are kept (no -b 1) to preserve the 64-D embeddings.
    """
    gdal_translate = require_cmd("gdal_translate")
    cleaned = []
    temp_dir.mkdir(parents=True, exist_ok=True)
    for idx, p in enumerate(tile_paths, start=1):
        out = temp_dir / p.name
        cmd = [
            gdal_translate,
            "-ot",
            "float32",
            "-co",
            "TILED=YES",
            "-co",
            "COMPRESS=NONE",
            "-co",
            "BIGTIFF=YES",
            str(p),
            str(out),
        ]
        log(f"[clean {idx}/{len(tile_paths)}] {p.name} -> {out.name}")
        ok = run(cmd, allow_fail=skip_errors)
        if ok:
            cleaned.append(out)
        elif skip_errors:
            log(f"Skipping tile due to translate failure: {p}")
        else:
            break
    return cleaned


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast mosaic AlphaEarth tiles (VRT + gdalwarp).")
    p.add_argument(
        "--year",
        type=int,
        default=2017,
        help="Year for default paths (default: 2017).",
    )
    p.add_argument(
        "--input-base",
        type=Path,
        default=None,
        help="Base path used for per-tile downloads. If omitted, defaults to map/<year>/alphaearth/brahmanbaria_alphaearth_<year>.tif",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GeoTIFF path. If omitted, defaults to map/<year>/alphaearth/brahmanbaria_alphaearth_<year>_mosaic.tif",
    )
    p.add_argument(
        "--gdal-cache-mb",
        type=int,
        default=4096,
        help="GDAL cache/warp memory in MB (default: 4096).",
    )
    p.add_argument(
        "--compress",
        type=str,
        default="NONE",
        help="GeoTIFF compression (default: NONE). Alternatives: ZSTD, LZW, DEFLATE.",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="GeoTIFF internal tile size (default: 256).",
    )
    p.add_argument(
        "--clean-tiles",
        action="store_true",
        help="Pre-clean tiles with gdal_translate to single-band float32 (helps with predictor/extrasamples issues).",
    )
    p.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip tiles that fail during cleaning (useful for occasional corrupt tiles).",
    )
    p.add_argument(
        "--keep-vrt",
        action="store_true",
        help="Keep the intermediate .vrt next to the output (default: delete).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists (default: fail if output exists).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    input_base = args.input_base or Path(f"map/{args.year}/alphaearth/brahmanbaria_alphaearth_{args.year}.tif")
    output = args.output or Path(f"map/{args.year}/alphaearth/brahmanbaria_alphaearth_{args.year}_mosaic.tif")

    tile_paths = list_tile_paths(input_base)
    log(f"Found {len(tile_paths)} tiles.")

    if args.clean_tiles:
        temp_dir = output.parent / "tmp_clean_tiles"
        log(f"Cleaning tiles into {temp_dir} ...")
        tile_paths = clean_tiles(tile_paths, temp_dir, skip_errors=args.skip_errors)

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")

    ref = read_reference(tile_paths)
    nodata = ref.get("nodata", 0)
    dtype = ref.get("dtype", "float32")
    # Force to float32 to avoid 64-bit predictor issues
    if dtype.lower() == "float64":
        dtype = "float32"
    try:
        ref_crs_wkt = ref["crs"].to_wkt() if ref.get("crs") else None
    except Exception:
        ref_crs_wkt = None

    vrt_path = output.with_suffix(output.suffix + ".vrt")

    log("Building VRT...")
    build_vrt(tile_paths, vrt_path, ref_crs_wkt=ref_crs_wkt)

    log("Warping VRT to final GeoTIFF (this is the main work)...")
    warp_to_geotiff(
        vrt_path,
        output,
        nodata=nodata,
        dtype=dtype,
        gdal_cache_mb=max(256, int(args.gdal_cache_mb)),
        compress=args.compress,
        tile_size=max(128, int(args.tile_size)),
        overwrite=bool(args.overwrite),
    )

    if not args.keep_vrt:
        try:
            vrt_path.unlink(missing_ok=True)
        except Exception:
            pass

    log(f"Saved mosaic to {output}")


if __name__ == "__main__":
    main()
