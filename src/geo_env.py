# src/geo_env.py
import os
import importlib.util
from pathlib import Path

# Locate rasterio package WITHOUT importing it
spec = importlib.util.find_spec("rasterio")
if spec is None or spec.origin is None:
    raise RuntimeError("rasterio is not installed")

rasterio_pkg_dir = Path(spec.origin).parent

proj_data_dir = rasterio_pkg_dir / "proj_data"
proj_db = proj_data_dir / "proj.db"

if not proj_db.exists():
    raise RuntimeError(f"proj.db not found at {proj_db}")

# Modern PROJ (>=9) requires PROJ_DATA
os.environ["PROJ_DATA"] = str(proj_data_dir)

# Backward compatibility (safe)
os.environ["PROJ_LIB"] = str(proj_data_dir)

# Remove poisoned GDAL_DATA
if "GDAL_DATA" in os.environ and not os.path.exists(os.environ["GDAL_DATA"]):
    os.environ.pop("GDAL_DATA", None)
