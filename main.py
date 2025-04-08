from utils.wtms import WMTSBuilder, WMTSMap

wmts_builder = WMTSBuilder("https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts")
maps = wmts_builder.build_maps()

# Coordinates EPSG:28992
c1 = (97168.5078, 417134.0288)
c2 = (97160.5078, 417126.0288)


image = maps[0].get_image(c1, c2)