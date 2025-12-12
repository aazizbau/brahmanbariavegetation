import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Visualize Sentinel-2 Bands\n",
                "Sample a random valid pixel from the clipped mosaic and visualize band values.\n",
                "\n",
                "- Default raster: `data/processed/mosaic_s2_oct_2016_clipped.tif` (resolved from project root).\n",
                "- Bands follow the download pipeline (B2, B3, B4, B5, B6, B7, B8, B8A).\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import rasterio\n",
                "from rasterio.windows import Window\n",
                "\n",
                "default_rel = Path('data/processed/mosaic_s2_oct_2016_clipped.tif')\n",
                "project_root = Path.cwd()\n",
                "if project_root.name == 'notebooks':\n",
                "    project_root = project_root.parent\n",
                "candidate = project_root / default_rel\n",
                "if candidate.exists():\n",
                "    raster_path = candidate\n",
                "else:\n",
                "    matches = list(project_root.rglob(default_rel.name))\n",
                "    if not matches:\n",
                "        raise FileNotFoundError(f'Could not locate {default_rel} from project root {project_root}')\n",
                "    raster_path = matches[0]\n",
                "\n",
                "print(f'Using raster: {raster_path}')\n",
                "print(f'Current working dir: {Path.cwd()}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def sample_random_valid_pixel(src, attempts=1000):\n",
                "    for _ in range(attempts):\n",
                "        row = np.random.randint(0, src.height)\n",
                "        col = np.random.randint(0, src.width)\n",
                "        mask = src.read_masks(1, window=Window(col, row, 1, 1))\n",
                "        if mask[0, 0] == 0:\n",
                "            continue  # nodata pixel\n",
                "        values = src.read(src.indexes, window=Window(col, row, 1, 1))[:, 0, 0]\n",
                "        x, y = src.transform * (col + 0.5, row + 0.5)\n",
                "        return (row, col), (x, y), values\n",
                "    raise ValueError('No valid pixel found after multiple attempts; check your raster/mask.')\n",
                "\n",
                "with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False, GDAL_PAM_ENABLED=False):\n",
                "    with rasterio.open(raster_path) as src:\n",
                "        (row, col), (x, y), values = sample_random_valid_pixel(src)\n",
                "        band_labels = list(src.descriptions) if src.descriptions and any(src.descriptions) else ['B2','B3','B4','B5','B6','B7','B8','B8A'][: len(src.indexes)]\n",
                "\n",
                "print(f'Random valid pixel at row={row}, col={col}, coords=({x:.5f}, {y:.5f})')\n",
                "print('Band values:')\n",
                "for label, val in zip(band_labels, values):\n",
                "    print(f'  {label}: {val}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 4))\n",
                "plt.bar(range(len(values)), values, tick_label=band_labels)\n",
                "plt.title('Band values at random valid pixel')\n",
                "plt.ylabel('Reflectance (scaled DN)')\n",
                "plt.xlabel('Band')\n",
                "plt.xticks(rotation=45)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

nb_path = Path('notebooks/visualize_bands.ipynb')
nb_path.write_text(json.dumps(nb, indent=2))
print(f"Wrote {nb_path}")
