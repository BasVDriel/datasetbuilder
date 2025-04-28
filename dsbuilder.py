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
ahn_dem_url = "https://basisdata.nl/hwh-ahn/ahn4/02a_DTM_0.5m/M_{tile}.zip" # tile must be capitals
ahn_dtm_url = "https://basisdata.nl/hwh-ahn/ahn4/03a_DSM_0.5m/R_{tile}.zip"
ahn_pntcloud_url = "https://geotiles.citg.tudelft.nl/AHN5_T/{tile}.LAZ"
orthomosaic_wmts_url = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts"

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
            if yn.lower() == 'y':
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
    
    def wmts_quality(self, name):
        from utils.wtms import WMTSBuilder
        from sewar import vifp
        import matplotlib.pyplot as plt
        import time

        time_taken = []
        image_paths = []
        vifp_values = []
        for detail in range(10, 20):
            wmts_builder = WMTSBuilder(orthomosaic_wmts_url, detail=detail)
            maps = wmts_builder.build_maps()
            # Coordinates EPSG:28992
            c1 = (86404.199,448706.490)
            c2 = (c1[0] + 50, c1[1] - 50)

            path = f"temp/output_{detail}.tif"
            t1 = time.time()
            image = maps[0].get_image(c1, c2)
            print(image)
            maps[0].save_image(image, c1, c2, path=path)
            t2 = time.time()

            time_taken.append(t2 - t1)
            image_paths.append(path)
            print(f"Detail {detail} took {time_taken[-1]} seconds")
        
        ref = plt.imread(image_paths[-1])
        for path in image_paths:
            print(path)
            image = plt.imread(path)
            vifp_val = vifp(ref, image)
            vifp_values.append(vifp_val)

        # Plot the time taken and VIFP values
        print(time_taken)
        print(vifp_values)
        # plot the quality level per point
        plt.scatter(time_taken, vifp_values, label="quality vs time", marker="o")
        for i, txt in enumerate(range(10, 20)):
            plt.annotate(txt, (time_taken[i], vifp_values[i]), fontsize=8, ha='right', va='bottom')            
        plt.legend()
        plt.xlabel('Time taken (s)')
        plt.ylabel('VIFP')
        plt.title('VIFP vs Time taken')
        plt.show()
        plt.imsave(name)
        


    def get_tiles(self, plot=True):
        """
        Downloads the tiles and subtiles for the AHN data
        """
        import pandas as pd
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from tqdm import tqdm
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
            ax = result_tile_polygons.plot(color='blue', alpha=0.5, edgecolor='black', label="Tiles")
            result_subtile_polygons.plot(ax=ax, color='green', alpha=0.5, edgecolor='black', label="Subtiles")
            filtered_trees_gdf.plot(ax=ax, color="red", marker=".")
            plt.show()

        # check for user input
        n_tiles = len(result_tile_polygons)
        n_subtiles = len(result_subtile_polygons)
        yn = input(f"Do you want to download {n_tiles} tiles and {n_subtiles} subtiles? (y/n): ")
        if yn.lower() != 'y':
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

if __name__ == '__main__':
    fire.Fire(DSBuilder)

