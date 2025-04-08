from utils.wtms import WMTSBuilder, WMTSMap
import time


for detail in range(15,19):
    wmts_builder = WMTSBuilder("https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts", detail=detail)
    maps = wmts_builder.build_maps()

    # Coordinates EPSG:28992
    c1 = (85742.87,448572.73)
    c2 = (85883.69,448486.15)

    t1 = time.time()
    image = maps[0].get_image(c1, c2)
    maps[0].save_image(image, c1, str(detail) + ".tif")
    t2 = time.time()
    print(detail, t2-t1)