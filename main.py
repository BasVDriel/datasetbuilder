import fire
import os

source_dir = "sources"
ahn_tile_fn = "ahn_subtiles.zip"
ahn_subtile_fn = "ahn_tiles.zip"
utrecht_trees_fn = "utrecht_trees.gpkg"

ahn_subtile_index = os.path.join(source_dir, ahn_subtile_fn)
ahn_tile_index = os.path.join(source_dir, ahn_tile_fn)
utrecht_boomkaart = os.path.join(source_dir, utrecht_trees_fn)

utrecht_trees_url = "https://arcgis.com/sharing/rest/content/items/7e2404cf7fba4bb087935f9cdb51f053/data"
ahn_subtile_url = "https://static.fwrite.org/2023/01/AHN_subunits_GeoTiles.zip"
ahn_tile_url = "https://static.fwrite.org/2023/01/AHN_AHN_GeoTiles.zip"
orthomosaic_wmts_url = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts"

def get_sources():
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

def ahn_test():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from utils.tiles import TileDownloader

    # Paths
    ahn_tiles_path = "zip://" + ahn_tile_index
    geojson_path = utrecht_boomkaart
    gdf_polygons = gpd.read_file(ahn_tiles_path)
    gdf_points = gpd.read_file(geojson_path)
    em_polygons = gpd.read_file(elevation_model_index)

    # Joint the points and polygons that contain a tree from the utrecht dataset
    joined = gpd.sjoin(gdf_polygons, gdf_points, how="inner", predicate="contains")
    result_polygons = gdf_polygons[gdf_polygons.index.isin(joined.index)]

    print(result_polygons.columns)
    print(result_polygons["GT_AHNSUB"])
    print(result_polygons["GT_AHN"])



    


if __name__ == '__main__':
    fire.Fire()

