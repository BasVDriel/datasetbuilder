

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

  * `dataset.geojson` The Geojson dataset has the following attributes as described below. Most of them have been carried over from the Utrecht dataset so they are not new columns. Those have been marked with  (carryover from Utrecht).
	  * **Nederlandse_naam** - Species name in dutch (carryover from Utrecht)
	  * **Wetenschappelijke_naam** - Scientific species name, this is used as a label  (carryover from Utrecht)
	  * **Plantjaar** - Year in which the tree was planted  (carryover from Utrecht)
	  * **Eigenaar** - The owner that has jurisdiction over the tree  (carryover from Utrecht)
	  * **Buurt** - Neighbourhood in which the tree exists  (carryover from Utrecht)
	  * **Wijk** - District in which the tree exists  (carryover from Utrecht)
	  * **Boomnummer** - A unique number for the tree within the Utrecht database  (carryover from Utrecht)
	  * **Zeldzaam** - Whether the tree is rare. "ja" or "nee" (carryover from Utrecht) 
	  * **Oud** - Whether the tree is older than 80 years. "ja" or "nee"   (carryover from Utrecht)
	  * **Bospltsn** Whether a tree is in a part. 1 or 0  (carryover from Utrecht)
	  * **Legenda** The genus of the tree  (carryover from Utrecht)
	  * **Exportdat** The data that the tree was entered into the Utrecht database in yyyymmdd  (carryover from Utrecht)
	  * **x** The X component of the stem coordinate in EPSG:28992 
	  * **y** The Y component of the stem coordinate in EPGS:28992
	  * **yyyy_gsdnn** The path to the specific orthomosaic directory in the file structure, where the resolution (nn) can be 8 cm or 25 cm. The year (yyyy) is specified in four digits. There can be multiple of these
	  * **las_path** The path to the .las file in the dataset.
	  * **Sentinel_path** The path to the .nc file in the dataset.
	  * **geometry** The canopy polygon. Interpretable by many geojson readers. Can be read out with shapely in python.

The dataset is organized with subdirectories for each data modality (point clouds, orthomosaics, Sentinel images) facilitating easy access and future processing.

---

## Data description
### Sources
The data in this dataset is sourced from various public sources, which can be accessed through their respective APIs.
- The AHN tiles are sourced from geotiles at https://geotiles.citg.tudelft.nl/
- The Orthophotos are sourced from PDOK at https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?&request=GetCapabilities&service=WMTS
- The Sentinel-2 data are sourced by Planetary Computer at https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a

### Data types
- **Sentinel** - The spectral data cubes from sentinel 2 are stored in the NetCDF (.nc) format. Specific descriptions regarding cell size, wavelength and such can be found in the files themselves.
- **Point cloud** - The point clouds are stored in the (.las) format. It is difficult to find the specification all of the attributes, but here are the ones that I am certain of
	- **X**  - X component of coordinate
	- **Y** - Y component of coordinate
	- **Z** - Z component of coordinate
	- **intensity** - Returned intensity of the LIDAR pulse
	- **return_number** - The n-th reflection from the lidar pulse
	- **number_of_returns** - Number of reflections from the LIDAR pulse (e.g. from foliage penetration)
	- **synthetic** - Unclear
	- **key_point** - Unclear
	- **withheld** - Unclear
	- **overlap** - Unclear
	- **scanner_channel** - The laser channel on the LIDAR that the point came from
	- **scan_direction_flag** - Unclear
	- **edge_of_flight_line** - Probably whether the point is in the edge of a flight line.
	- **classification** - The class as specified by https://www.ahn.nl/kwaliteitsbeschrijving
	- **user_data** - Unclear
	- **scan_angle** - Unclear
	- **point_source_id** - Unclear
	- **gps_time** - Likely the GPS time of the sensor.
	- **red** - Red channel from orthophoto inpainting
	- **green** - Green channel from orthophoto inpainting
	- **blue** - Blue channel from orthophoto inpainting
	- **nir** - Infrared channel from orthophoto inpainting, likely from the CIR imagery.
- **Orthomosaics** - The orthomosaics are stored as geotiff files. The files have their CRS implicitly encoded (EPSG:28992). Other than than, the files just have three bands R, G and B.

## License

T.B.D.

