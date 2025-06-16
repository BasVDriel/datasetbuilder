

# Dataset Builder (DSBuilder)

A Python tool to download, process, and build datasets from AHN (Dutch elevation data), Sentinel-2 satellite imagery, and Utrecht tree data. This tool supports tile-based downloading, canopy height model generation, tree segmentation, and dataset construction with polygonal tree crowns.


## Installation

1. Clone the repository.
2. Install the packages
- Python 3.12.10
- conda dsbuilder create -f environment.yml
- conda activate dsbuilder

---

## Setup

By default, the script creates a symbolic link from `/mnt/datapart/datasetbuilder/` to the local `sources/` directory for managing large datasets. This can be changed in dsbuilder.py

---

## Usage

The `DSBuilder` class exposes its methods via CLI using `fire`. Run commands as:

```bash
python dsbuilder.py <method> [--arg1 value1 --arg2 value2 ...]
```

### Available commands

* `get_sources`: Downloads base datasets required for tile downloading (Utrecht trees, AHN tile shapefiles).
* `get_tiles`: Downloads AHN tiles (DEM, DTM, point clouds) and Sentinel-2 tiles covering trees. Optionally plots tile coverage.
* `wmts_quality`: Evaluates the quality vs download time of WMTS orthomosaic images across different detail levels. Supports downloading and plotting.
* `chm_test`: Generates and visualizes a Canopy Height Model (CHM) for a sample AHN point cloud tile.
* `build_dataset`: Constructs a dataset for a given subtile, segments trees, generates polygons, downloads orthomosaics, and saves outputs to `/output`.

---

## Examples

1. **Download Utrecht tree database and AHN tile structure:**

```bash
python dsbuilder.py get_sources
```

2. **Download all AHN and Sentinel tiles that cover Utrecht trees, with plot:**

```bash
python dsbuilder.py get_tiles --plot True
```

3. **Build dataset for a single subtile (e.g., `31HN1_20`):**

```bash
python dsbuilder.py build_dataset --subtile_idx "31HN1_20"
```

4. **Run CHM test on sample tile:**

```bash
python dsbuilder.py chm_test
```

5. **Evaluate WMTS orthomosaic quality and download images:**

```bash
python dsbuilder.py wmts_quality --exp_name "test_exp" --download True --plot True
```

---

## Output Structure

* `/sources`: Contains downloaded base data (tiles, Utrecht trees, etc.)
* `/ahn_tiles`: AHN point cloud tiles downloaded
* `/sentinel`: Sentinel-2 tiles downloaded
* `/output`:

  * `dataset.geojson`: GeoJSON dataset with enriched attributes:

    * Tree scientific name
    * Tree coordinates (`x`, `y`)
    * Paths to orthomosaic images (high and low resolution)
    * Paths to point cloud and Sentinel tiles
    * Canopy polygon geometry for each tree

The dataset is organized with subdirectories for each data modality (point clouds, orthomosaics, Sentinel images) facilitating easy access and future processing.

---


## License

T.B.D.

