import fire

ahn_tile_index = "sources/AHN_subunits_GeoTiles.zip"
orthomosaic_wmts_url = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts"
utrecht_boomkaart = "sources/trees_U.geojson"
elevation_model_index = "sources/kaartbladindex.json"

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

    # Paths
    ahn_tiles_path = "zip://" + ahn_tile_index
    geojson_path = utrecht_boomkaart
    gdf_polygons = gpd.read_file(ahn_tiles_path)
    gdf_points = gpd.read_file(geojson_path)
    em_polygons = gpd.read_file(elevation_model_index)

    # Joint the points and polygons that contain a tree from the utrecht dataset
    joined = gpd.sjoin(gdf_polygons, gdf_points, how="inner", predicate="contains")
    result_polygons = gdf_polygons[gdf_polygons.index.isin(joined.index)]
    em_polygons.plot()
    plt.show()



if __name__ == '__main__':
    fire.Fire()

