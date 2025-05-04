import fire
import os

source_dir = "sources"
dem_dir = os.path.join(source_dir, "dem")
dtm_dir = os.path.join(source_dir, "dtm")
ahn_dir = os.path.join(source_dir, "ahn_tiles")

ahn_tile_fn = "ahn_subtiles.zip"
ahn_subtile_fn = "ahn_tiles.zip"
utrecht_trees_fn = "utrecht_trees.gpkg"

ahn_subtile_path = os.path.join(source_dir, ahn_subtile_fn)
ahn_tile_path = os.path.join(source_dir, ahn_tile_fn)
utrecht_trees_path = os.path.join(source_dir, utrecht_trees_fn)

utrecht_trees_url = "https://arcgis.com/sharing/rest/content/items/7e2404cf7fba4bb087935f9cdb51f053/data"
ahn_subtile_url = "https://static.fwrite.org/2023/01/AHN_subunits_GeoTiles.zip"
ahn_tile_url = "https://static.fwrite.org/2023/01/AHN_AHN_GeoTiles.zip"
ahn_dem_url = "https://basisdata.nl/hwh-ahn/ahn4/02a_DTM_0.5m/M_{tile}.zip" 
ahn_dtm_url = "https://basisdata.nl/hwh-ahn/ahn4/03a_DSM_0.5m/R_{tile}.zip"
ahn_pntcloud_url = "https://geotiles.citg.tudelft.nl/AHN5_T/{tile}.LAZ"
orthomosaic_wmts_url = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts"
sentinel2_url = "https://planetarycomputer.microsoft.com/api/stac/v1"

class DSBuilder:
    def __init__(self):
        pass

    def clean_sources(self, path=source_dir, test=False):
        """
        Gives the option to delete the sources directory
        """
        from utils.tiles import delete_recursively
        if os.path.exists(path):
            print(f"Do you want to delete the contents of {path} recursively? (y/n): ")
            yn = input()
            if yn.lower() == "y":
                delete_recursively(path, test=True)
        else:
            print(f"Directory {path} does not exist. No action taken.")

    def get_sources(self):
        """
        Downloads key sources that are required for tile downloading
        """
        from utils.tiles import download_url
        print("Downloading sources from source urls")
        download_url(ahn_subtile_url, source_dir, filename=ahn_subtile_fn)
        download_url(ahn_tile_url, source_dir, filename=ahn_tile_fn)
        download_url(utrecht_trees_url, source_dir, filename=utrecht_trees_fn)
    
    def wmts_quality(self, exp_name, download=False, plot=False):
        from utils.wtms import WMTSBuilder
        from tqdm import tqdm
        from sewar import vifp
        import matplotlib.pyplot as plt
        import time
        import pandas as pd

        exp_entries = []
        detail_range = range(10, 20)
        csv_file = f"vif_experiments.csv"

        # Download images if specified
        if download:
            for detail in tqdm(detail_range, desc="Downloading images"):
                wmts_builder = WMTSBuilder(orthomosaic_wmts_url, detail=detail)
                maps = wmts_builder.build_maps()
                # Coordinates EPSG:28992
                c1 = (127677.90,431678.96)
                test_size = 25
                c2 = (c1[0] + test_size, c1[1] - test_size)

                t1 = time.time()
                image = maps[0].get_image(c1, c2)
                t2 = time.time()
                tdiff = t2 - t1
                path = f"temp/{exp_name}_{detail}_{int(tdiff * 1000)}.tif"
                maps[0].save_image(image, c1, c2, path=path)

                exp_entries.append((int(tdiff * 1000), detail, path))  # Store time in ms, detail, and path
                tqdm.write(f"Detail {detail} took {int(1000 * tdiff)} ms")

            # Sort by detail level
            exp_entries = sorted(exp_entries, key=lambda entry: entry[1])

            # Use the image of max detail for reference
            ref_image_path = exp_entries[-1][2]  # Path of the highest detail image
            ref_image = plt.imread(ref_image_path)

            # Calculate VIFP values for each entry
            vifp_values = []
            for time_ms, detail, path in tqdm(exp_entries, desc="Computing VIF"):
                tqdm.write(f"Processing {path}")
                image = plt.imread(path)
                vifp_val = vifp(ref_image, image)
                vifp_values.append(vifp_val)

                # Append data to CSV
                with open(csv_file, 'a') as f:
                    f.write(f"{time_ms},{vifp_val},{detail},{test_size},{exp_name}\n")

        # If plotting is enabled, load the CSV and plot the results
        if plot:
            df = pd.read_csv(csv_file, header=None)
            df.columns = ["Time (ms)", "VIFP Quality", "Detail Level", "Image Size (m)", "Experiment"]
            df = df[df["Experiment"] == exp_name]  # Filter by experiment name
            time_values = df["Time (ms)"].to_numpy() / 1000  # Convert milliseconds to seconds
            vifp_values = df["VIFP Quality"].to_numpy()
            detail_levels = df["Detail Level"].to_numpy()

            # Create a colormap
            cmap = plt.get_cmap("viridis", len(detail_range))  # Use a colormap with discrete colors
            plt.scatter(time_values, vifp_values, c=detail_levels, cmap=cmap, marker="o")

            # Create a legend
            handles = []
            for i in range(len(detail_range)):
                handles.append(plt.Line2D([0], [0], marker='o', color='w', label=str(detail_range.start + i),
                    markerfacecolor=cmap(i), markersize=10))
            plt.legend(handles=handles, title="Detail Level", bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout(pad=3)
            plt.xlim(0.05,100)
            plt.xlabel("Time (s)")
            plt.ylabel("VIFP Quality")
            plt.xscale("log")
            plt.title(f"WMTS Quality for {exp_name}")
            plt.grid(which='both')
            plt.savefig(f"{exp_name}_quality_plot.png")  
            plt.show()


    def get_sentinel(self):
        from pystac_client import Client
        import planetary_computer
        from odc.stac import load
        import xarray as xr
        import matplotlib.pyplot as plt
        from pyproj import Transformer 

        # Define the bounding box in EPSG:28992 (RD New)
        x, y = 92762.52, 420165.06
        buffer = 50
        bbox_rd = (x - buffer, y - buffer, x + buffer, y + buffer)

        # Convert RD New (EPSG:28992) to WGS84 (EPSG:4326)
        transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
        min_lon, min_lat = transformer.transform(bbox_rd[0], bbox_rd[1])
        max_lon, max_lat = transformer.transform(bbox_rd[2], bbox_rd[3])
        bounds = (min_lon, min_lat, max_lon, max_lat)

        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bounds,
            datetime="2023-06-01/2023-08-30",
            query={"eo:cloud_cover": {"lt": 5}},
            max_items=5
        )

        items = list(search.get_items())
        print(f"Found {len(items)} items")

        # Sign items
        signed_items = [planetary_computer.sign(item) for item in items]

        # Load bands
        all_bands = [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
            "B09", "B11", "B12", "SCL", "AOT", "WVP"
        ]

        ds = load(
            signed_items,
            bands=all_bands,
            bbox=bounds,
            crs="EPSG:28992",  
            resolution=10,
            chunks={},
            groupby="solar_day",
        )

        # Visualize RGB
        ds_one = ds.isel(time=0)
        r = ds_one["B04"]
        g = ds_one["B03"]
        b = ds_one["B02"]

        rgb = xr.concat([r, g, b], dim="band").transpose("y", "x", "band")
        rgb /= rgb.max()

        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.axis("off")
        plt.title("Sentinel-2 RGB (B04-B03-B02)")
        plt.show()

        

    def get_tiles(self, plot=True):
        """
        Downloads the tiles and subtiles for the AHN data
        """
        import pandas as pd
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from utils.tiles import TileDownloader

        ahn_tiles_path = "zip://" + ahn_tile_path
        tile_polygons_gdf = gpd.read_file(ahn_tiles_path)
        subtile_polygons_gdf = gpd.read_file(ahn_subtile_path)

        # merge the two layers in the geopackage of the utrecht trees
        utrecht_trees_bos_gdf = gpd.read_file(utrecht_trees_path, layer=0)
        utrecht_trees_weg_gdf = gpd.read_file(utrecht_trees_path, layer=1)
        utrecht_trees_gdf = pd.concat([utrecht_trees_bos_gdf, utrecht_trees_weg_gdf], axis=0)
        
        # test on a single tree sample for now
        filtered_trees_gdf = utrecht_trees_gdf

        # Joint the points and polygons that contain a tree from the utrecht dataset
        joined = gpd.sjoin(tile_polygons_gdf, filtered_trees_gdf, how="inner", predicate="contains")
        result_tile_polygons = tile_polygons_gdf[tile_polygons_gdf.index.isin(joined.index)]

        # Also find the subtiles that contain a tree
        joined_subtiles = gpd.sjoin(subtile_polygons_gdf, filtered_trees_gdf, how="inner", predicate="contains")
        result_subtile_polygons = subtile_polygons_gdf[subtile_polygons_gdf.index.isin(joined_subtiles.index)]

        # aplot the result polygons for tiles and subtiles
        if plot:
            ax = result_tile_polygons.plot(color="blue", alpha=0.5, edgecolor="black", label="Tiles")
            result_subtile_polygons.plot(ax=ax, color="green", alpha=0.5, edgecolor="black", label="Subtiles")
            filtered_trees_gdf.plot(ax=ax, color="red", marker=".")
            plt.show()

        # check for user input
        n_tiles = len(result_tile_polygons)
        n_subtiles = len(result_subtile_polygons)
        yn = input(f"Do you want to download {n_tiles} tiles and {n_subtiles} subtiles? (y/n): ")
        if yn.lower() != "y":
            print("Download cancelled.")
            return
        
        #download a dem tiles
        print("\nDownloading dem tiles")
        dem_tile_dl = TileDownloader(url_template=ahn_dem_url, output_dir=dem_dir)
        for index, tile in result_tile_polygons.iterrows():
            tile_idx = tile["GT_AHN"]
            dem_tile_dl.download_tile(tile=tile_idx, file_name=tile_idx, unzip=True)

        # download  dtm tiles
        print("\nDownloading dtm tiles")
        ahn_subtile_dl = TileDownloader(url_template=ahn_dtm_url, output_dir=dtm_dir)
        for index, tile in result_tile_polygons.iterrows():
            tile_idx = tile["GT_AHN"]
            ahn_subtile_dl.download_tile(tile=tile_idx, file_name=tile_idx, unzip=True)

        # download the point cloud tiles
        print("\nDownloading point cloud tiles")
        ahn_pntcloud_dl = TileDownloader(url_template=ahn_pntcloud_url, output_dir=ahn_dir)
        for index, tile in result_subtile_polygons.iterrows():
            tile_idx = tile["GT_AHNSUB"]
            ahn_pntcloud_dl.download_tile(tile=tile_idx, file_name=tile_idx, unzip=True)

if __name__ == "__main__":
    fire.Fire(DSBuilder)

