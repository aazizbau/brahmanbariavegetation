# Brahmanbaria Vegetation Change Study

Research workspace for analyzing temporal vegetation changes in the Brahmanbaria district of Bangladesh using spectral indices (e.g., NDVI, EVI, SAVI). The repository is structured for downloading satellite imagery, mosaicing scenes, computing indices, and summarizing trends.

## Project layout
- `data/raw/` — downloaded scenes per date/tile (keep unprocessed data here).
- `data/interim/` — reprojected/clipped/mosaiced rasters before index calculation.
- `data/processed/` — final index rasters, composites, and statistics tables.
- `data/external/` — ancillary data such as AOI boundaries or masks.
- `notebooks/` — exploratory notebooks; prefer numbered naming (e.g., `01_download.ipynb`).
- `src/` — reusable code:
  - `src/data/` — data access, tiling, and AOI utilities.
  - `src/indices/` — spectral index calculations and mosaicing helpers.
  - `src/utils/` — shared helpers (logging, config, io).
- `docs/` — notes and methodology.
- `reports/figures/` — exported plots/maps for reporting.

## Getting started
1) Create and activate the virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
```

2) Install dependencies (ensure system GDAL dependencies are available for `rasterio`/`geopandas`):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Configure environment variables if needed (e.g., `GOOGLE_APPLICATION_CREDENTIALS`, Earth Engine service account keys). Use a local `.env` that is already git-ignored.

## Suggested workflow
1) **Define AOI**: store Brahmanbaria boundary as GeoJSON/Shapefile in `data/external/`.
2) **Download scenes**: use `sentinelsat` (Sentinel-2) or `earthengine-api` to pull images to `data/raw/` by date range and cloud cover.
3) **Preprocess & mosaic**: reproject to a common CRS, clip to AOI, cloud-mask, and mosaic by date; save to `data/interim/`.
4) **Compute indices**: derive NDVI/EVI/SAVI and temporal composites; save rasters to `data/processed/`.
5) **Analyze trends**: aggregate by season/year, generate plots/maps in notebooks, and export figures to `reports/figures/`.

## Earth Engine helpers (examples)
- Download tiled monthly composite (local GeoTIFFs, tiles sized in km):
  ```bash
  python src/data/download_gee_s2_composite.py --month october --year 2016 --collection l1c --scale 10 --tile-width-km 5 --tile-height-km 5 --tile-overlap-km 0.5 --output-dir data/raw/gee/tiles
  ```
- Check which tiles are present/missing:
  ```bash
  python src/data/check_download_gee_s2_composite.py --month october --year 2016 --tile-width-km 5 --tile-height-km 5 --tile-overlap-km 0.5 --output-dir data/raw/gee/tiles
  ```
- Download only missing tiles:
  ```bash
  python src/data/download_missing_gee_s2_composite.py --month october --year 2016 --collection l1c --scale 10 --tile-width-km 5 --tile-height-km 5 --tile-overlap-km 0.5 --output-dir data/raw/gee/tiles
  ```
- Mosaic downloaded tiles into a single GeoTIFF:
  ```bash
  python src/preprocessing/mosaic_s2_composite.py --input-dir data/raw/gee/tiles --prefix s2_oct_2016 --year 2016 --month 10 --output data/interim/mosaic_s2.tif
  ```

## Notes
- Keep large rasters out of version control; `.gitignore` preserves folder structure via `.gitkeep`.
- Prefer reproducible notebooks with clear parameters; extract reusable code into `src/`.
- Add a `config.yaml` later if you want to centralize AOI paths, date ranges, and band mappings.
