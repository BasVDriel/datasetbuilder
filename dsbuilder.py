import fire
import os

source_dir = "sources"
linked_source_dir = "/mnt/datapart/datasetbuilder/"

# set up symbolic link to the large disk if it has not been done yet
try:
    os.symlink(linked_source_dir, source_dir, target_is_directory=True)
except FileExistsError:
    pass

dem_dir = os.path.join(source_dir, "dem")
dtm_dir = os.path.join(source_dir, "dtm")
ahn_dir = os.path.join(source_dir, "ahn_tiles")
sentinel_dir = os.path.join(source_dir, "sentinel")

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

    def clean_sources(self, path=source_dir, test=True):
        """
        Gives the option to delete the sources directory
        """
        from utils.download import delete_recursively
        if os.path.exists(path):
            print(f"Do you want to delete the contents of {path} recursively? (y/n): ")
            yn = input()
            if yn.lower() == "y":
                delete_recursively(path, test=test)
        else:
            print(f"Directory {path} does not exist. No action taken.")

    def get_sources(self):
        """
        Downloads key sources that are required for tile downloading
        """
        from utils.download import download_url
        print("Downloading sources from source urls")
        download_url(ahn_subtile_url, source_dir, filename=ahn_subtile_fn)
        download_url(ahn_tile_url, source_dir, filename=ahn_tile_fn)
        download_url(utrecht_trees_url, source_dir, filename=utrecht_trees_fn)
    
    def wmts_quality(self, exp_name, download=False, plot=False):
        from utils.download import WMTSBuilder
        from tqdm import tqdm
        from sewar import vifp
        import matplotlib.pyplot as plt
        import pandas as pd

        exp_entries = []
        detail_range = range(10, 20)
        csv_file = f"vif_experiments.csv"

        # Download images if specified
        if download:
            for detail in tqdm(detail_range, desc="Downloading images"):
                wmts_builder = WMTSBuilder(source_dir, orthomosaic_wmts_url, detail=detail)
                maps = wmts_builder.build_maps()

                # Coordinates EPSG:28992
                c1 = (127677.90,431678.96)
                test_size = 25
                c2 = (c1[0] + test_size, c1[1] - test_size)

                image, tdiff = maps[0].get_image(c1, c2, timing=True)
                image_name = f"{exp_name}_{detail}_{int(tdiff * 1000)}.tif"
                image_path = maps[0].save_image(c1, c2, file_name=image_name, img=image)

                exp_entries.append((int(tdiff * 1000), detail, image_path))  # Store time in ms, detail, and path
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
  

    def get_tiles(self, plot=True, tree_idx=None):
        """
        Downloads the tiles and subtiles for the AHN data
        """
        import pandas as pd
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from utils.download import TileDownloader
        from utils.download import SentinelDownloader

        ahn_tiles_path = "zip://" + ahn_tile_path
        tile_polygons_gdf = gpd.read_file(ahn_tiles_path)
        subtile_polygons_gdf = gpd.read_file(ahn_subtile_path)

        # merge the two layers in the geopackage of the utrecht trees
        utrecht_trees_bos_gdf = gpd.read_file(utrecht_trees_path, layer=0)
        utrecht_trees_weg_gdf = gpd.read_file(utrecht_trees_path, layer=1)
        utrecht_trees_gdf = pd.concat([utrecht_trees_bos_gdf, utrecht_trees_weg_gdf], axis=0)
        
        # test on a single tree sample for now
        if tree_idx:
            filtered_trees_gdf = utrecht_trees_gdf.iloc[[tree_idx]]
        else:
            filtered_trees_gdf = utrecht_trees_gdf

        # Joint the points and polygons that contain a tree from the utrecht dataset
        joined = gpd.sjoin(tile_polygons_gdf, filtered_trees_gdf, how="inner", predicate="contains")
        result_tile_polygons = tile_polygons_gdf[tile_polygons_gdf.index.isin(joined.index)]

        # Also find the subtiles that contain a tree
        joined_subtiles = gpd.sjoin(subtile_polygons_gdf, filtered_trees_gdf, how="inner", predicate="contains")
        result_subtile_polygons = subtile_polygons_gdf[subtile_polygons_gdf.index.isin(joined_subtiles.index)]

        # aplot the result polygons for tiles and subtiles
        if plot:
            fig, ax = plt.subplots(figsize=(3.94*2, 3.94*2))
            result_tile_polygons.plot(ax=ax, color="tab:blue", edgecolor="black", label="Tiles")
            result_subtile_polygons.plot(ax=ax, color="tab:orange", edgecolor="black", label="Subtiles")
            filtered_trees_gdf.plot(ax=ax, color="red", marker="+")
            # Manually add legend entries for Tiles, Subtiles, and Trees
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Tiles',
                       markerfacecolor='tab:blue', markersize=10, markeredgecolor='black'),
                Line2D([0], [0], marker='o', color='w', label='Subtiles',
                       markerfacecolor='tab:orange', markersize=10, markeredgecolor='black'),
                Line2D([0], [0], marker='+', color='red', label='Trees', markersize=10, linestyle='None')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            ax.set_xlabel("Easting")
            ax.set_ylabel("Northing")
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

        # download sentinel 2 datacubes as tiles
        print("\nDownloading Sentinel 2 tiles")
        sentinel_dl = SentinelDownloader(url=sentinel2_url, output_dir=sentinel_dir)
        for index, tile in result_subtile_polygons.iterrows():
            tile_idx = tile["GT_AHNSUB"]
            tile_bounds = tile["geometry"].bounds
            sentinel_dl.download_tile(tile_bounds=tile_bounds, tile_name=tile_idx)

    def chm_test(self):
        from utils.compute import pointcloud_to_chm
        import matplotlib.pyplot as plt
        import rasterio

        resolution = 0.5  # raster resolution in meters
        subtile_idx = "31HN1_20"
        las_file_path = os.path.join(ahn_dir, f"{subtile_idx}.LAZ")

        chm, transform, stats = pointcloud_to_chm(las_file_path, resolution)

        dem_stat, dsm_stat = stats

        cm=1/2.54
        plt.figure(figsize=(10*cm, 8*cm))  # Set a large figure size
        plt.hist(dem_stat.ravel(), bins=20, edgecolor='black')
        plt.xlim(1,10)
        plt.xlabel("Points per cell")
        plt.ylabel("Counts")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # === Plot CHM ===
        cm=1/2.54
        plt.figure(figsize=(10*cm, 8*cm))  # Set a large figure size
        plt.imshow(chm)
        cbar = plt.colorbar(label="Canopy Height (m)")
        cbar.ax.tick_params(labelsize=8)
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.grid(visible=True, alpha=0.5)
        plt.show()

        print(f"CHM saved to: {os.path.abspath(chm_path)}")
        chm_path = "output_chm.tif"
        with rasterio.open(
            chm_path,
            "w",
            driver="GTiff",
            height=chm.shape[0],
            width=chm.shape[1],
            count=1,
            dtype=chm.dtype,
            crs="EPSG:28992",
            transform=transform
        ) as dst:
            dst.write(chm, 1)


    def build_dataset(self, subtile_idx="31HN1_20", plot=True, resolution=0.5):
        from utils.compute import pointcloud_to_chm, tree_marker_grid, filter_labels, compute_polygons, pixel_to_world, file_writer
        import geopandas as gpd
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.segmentation import watershed
        from skimage import measure
        from scipy import ndimage as ndi
        from shapely.geometry import Polygon

        sentinel_file_path = os.path.join(sentinel_dir, f"{subtile_idx}.nc")

        # load tree poinst from the utrecht trees geopackage
        utrecht_trees_path = "sources/utrecht_trees.gpkg"
        layer0 = gpd.read_file(utrecht_trees_path, layer=0)
        layer1 = gpd.read_file(utrecht_trees_path, layer=1)
        trees_gdf = gpd.GeoDataFrame(pd.concat([layer0, layer1], ignore_index=True), crs=layer0.crs)

        # Construct CHM
        las_file_path = os.path.join(ahn_dir, f"{subtile_idx}.LAZ")
        chm, transform, stats = pointcloud_to_chm(las_file_path, resolution)

        # extract the specific subtile polygon
        subtile_polygons_gdf = gpd.read_file(ahn_subtile_path)
        subtile_polygon = subtile_polygons_gdf[subtile_polygons_gdf["GT_AHNSUB"] == subtile_idx]

        # Clip the trees to the subtile polygon
        sub_trees_gdf = gpd.clip(trees_gdf, subtile_polygon)

        # extract the coordinates of the trees
        tree_coords = [(geom.x, geom.y) for geom in sub_trees_gdf.geometry]
        markers, markers_array = tree_marker_grid(chm, tree_coords, transform)

        # Ensure contiguous memory for the markers
        # ABSOLUTELY CRUCIAL do not remove this line if watershed is in use! 
        # Watershed segmentation depends on the memory layout which has to be coniguous.
        markers = np.ascontiguousarray(markers) 

        # Run watershed segmentation
        chm_nan_to_zero = np.nan_to_num(chm, nan=0.0)
        elevation = ndi.gaussian_filter(chm_nan_to_zero, sigma=1)
        labels = watershed(-elevation, markers, mask=chm_nan_to_zero > 0)
        regions = measure.regionprops(labels, chm)
        
        # filter out poorly segmented labels for now until improvements are made to the segmentation
        filtered_labels, tree_regions = filter_labels(labels, regions, markers, markers_array)
 
        # extract the polygons around each raster mask
        image_coord_polygons = compute_polygons(tree_regions)

        # convert polygons back to world coordinates
        world_coord_polygons = {}
        for label, polygon in image_coord_polygons.items():
            x = polygon[:, 0]
            y = polygon[:, 1]
            world_coord_tuple = pixel_to_world(transform, x, y)
            world_coord_polygons[label] = np.stack(world_coord_tuple, axis=-1)

        # convert to shapely polygons 
        polygons = [Polygon(polygon) for label, polygon in world_coord_polygons.items()]

        # Find the trees that have polygons and correct the index for gpd
        tree_indices = np.array(list(world_coord_polygons.keys()))
        tree_indices = tree_indices - 1
        trees_with_poly = sub_trees_gdf.iloc[tree_indices].copy()
        
        # Assign valid shapely Polygon objects to the geometry column, also exchange the point coordinates for a point column
        points_list = trees_with_poly["geometry"]
        points = np.array([[point.x, point.y] for point in points_list])
        trees_with_poly["x"] = points[:, 0]
        trees_with_poly["y"] = points[:, 1]
        trees_with_poly["geometry"] = polygons

        # segment and write the point 
        trees_with_poly = file_writer(las_file_path, sentinel_file_path, trees_with_poly, orthomosaic_wmts_url)
        trees_with_poly.to_file("output/dataset.geojson", driver="GeoJSON")


if __name__ == "__main__":
    fire.Fire(DSBuilder)

