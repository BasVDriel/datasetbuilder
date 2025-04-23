import fire
import os

source_dir = "sources"
ahn_tile_fn = "ahn_subtiles.zip"
ahn_subtile_fn = "ahn_tiles.zip"
utrecht_trees_fn = "utrecht_trees.gpkg"

ahn_subtile_path = os.path.join(source_dir, ahn_subtile_fn)
ahn_tile_path = os.path.join(source_dir, ahn_tile_fn)
utrecht_trees_path = os.path.join(source_dir, utrecht_trees_fn)

utrecht_trees_url = "https://arcgis.com/sharing/rest/content/items/7e2404cf7fba4bb087935f9cdb51f053/data"
ahn_subtile_url = "https://static.fwrite.org/2023/01/AHN_subunits_GeoTiles.zip"
ahn_tile_url = "https://static.fwrite.org/2023/01/AHN_AHN_GeoTiles.zip"
orthomosaic_wmts_url = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts"

def get_sources():
    """"
    Downloads some key sources like tile indices that are required for tile downloading
    """
    from utils.tiles import download_url
    print("Downloading sources from source urls")
    download_url(ahn_subtile_url, source_dir, filename=ahn_subtile_fn)
    download_url(ahn_tile_url, source_dir, filename=ahn_tile_fn)
    download_url(utrecht_trees_url, source_dir, filename=utrecht_trees_fn)

def wmts_test():
    from utils.wtms import WMTSBuilder, WMTSMap
    import time

    wmts_builder = WMTSBuilder(orthomosaic_wmts_url, detail=15)
    maps = wmts_builder.build_maps()
    # Coordinates EPSG:28992
    c1 = (130821.89,457922.12)
    c2 = (130925.86,457840.79)

    t1 = time.time()
    image = maps[0].get_image(c1, c2)
    maps[0].save_image(image, c1)
    t2 = time.time()
    print(t2-t1)

def tiling_test():
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt

    ahn_tiles_path = "zip://" + ahn_tile_path
    tile_polygons_gdf = gpd.read_file(ahn_tiles_path)
    subtile_polygons_gdf = gpd.read_file(ahn_subtile_path)

    # merge the two layers in the geopackage of the utrecht trees
    utrecht_trees_bos_gdf = gpd.read_file(utrecht_trees_path, layer=0)
    utrecht_trees_weg_gdf = gpd.read_file(utrecht_trees_path, layer=1)
    utrecht_trees_gdf = pd.concat([utrecht_trees_bos_gdf, utrecht_trees_weg_gdf], axis=0)
    
    # Joint the points and polygons that contain a tree from the utrecht dataset
    joined = gpd.sjoin(tile_polygons_gdf, utrecht_trees_gdf, how="inner", predicate="contains")
    result_tile_polygons = tile_polygons_gdf[tile_polygons_gdf.index.isin(joined.index)]

    # Also find the subtiles that contain a tree
    joined_subtiles = gpd.sjoin(subtile_polygons_gdf, utrecht_trees_gdf, how="inner", predicate="contains")
    result_subtile_polygons = subtile_polygons_gdf[subtile_polygons_gdf.index.isin(joined_subtiles.index)]

    # aplot the result polygons for tiles and subtiles
    ax = result_tile_polygons.plot(color='blue', alpha=0.5, edgecolor='black', label="Tiles")
    result_subtile_polygons.plot(ax=ax, color='green', alpha=0.5, edgecolor='black', label="Subtiles")
    plt.show()


    


if __name__ == '__main__':
    fire.Fire()

