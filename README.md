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
- Clip the mosaic to the Brahmanbaria AOI:
  ```bash
  python src/preprocessing/clip_mosaic_s2_composite.py --input data/interim/mosaic_s2_oct_2016.tif --vector map/brahmanbaria_gpkg.gpkg --output data/processed/mosaic_s2_oct_2016_clipped.tif
  ```
- Compute NDVI from the clipped mosaic:
  ```bash
  python src/indices/make_ndvi.py --input data/processed/mosaic_s2_oct_2016_clipped.tif --output data/processed/ndvi_mosaic_s2_oct_2016.tif
  ```
- Download a single date (tiled) Sentinel-2 scene:
  ```bash
  python src/data/download_gee_s2.py --date 2017-11-03 --scale 10 --tile-width-km 5 --tile-height-km 5 --tile-overlap-km 0.5 --output-dir data/raw/gee/s2_single
  ```
- Mosaic single-date tiles:
  ```bash
  python src/preprocessing/mosaic_s2.py --input-dir data/raw/gee/s2_single --prefix s2_20171103 --year 2017 --month 11 --day 3 --output data/interim/mosaic_s2_20171103.tif
  ```
- Clip single-date mosaic to AOI:
  ```bash
  python src/preprocessing/clip_s2.py --input data/interim/mosaic_s2_20171103.tif --vector map/brahmanbaria_gpkg.gpkg --output data/processed/mosaic_s2_20171103_clipped.tif
  ```
- Compute NDVI from separate band files:
  ```bash
  python src/indices/make_ndvi_image.py --year 2017 --date 20171103 --red map/2017/S2_20171103_B04_10m.tif --nir map/2017/S2_20171103_B08_10m.tif --output map/2017/ndvi_20171103.tif
  ```
- Compute EVI from separate band files:
  ```bash
  python src/indices/make_evi_image.py --year 2017 --date 20171103 --scale 1.0
  ```
- Clip NDVI/EVI to AOI:
  ```bash
  python src/indices/clip.py --input map/2017/ndvi_20171103.tif
  # or any map/<year>/<metric>_<yyyymmdd>.tif
  ```
- Mosaic local SAFE band tiles with SCL masking:
  ```bash
  python src/preprocessing/make_mosaic_image.py --year 2018 --band B02 --output map/2018/S2_20181128_B02_10m.tif
  ```

## Notes
- Keep large rasters out of version control; `.gitignore` preserves folder structure via `.gitkeep`.
- Prefer reproducible notebooks with clear parameters; extract reusable code into `src/`.
- Add a `config.yaml` later if you want to centralize AOI paths, date ranges, and band mappings.
