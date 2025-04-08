import requests
import re
import rasterio 
from lxml import etree
import numpy as np
from io import BytesIO
from PIL import Image

class WMTSMap:
    def __init__(self, url_template, gsd, scale, tile_width, tile_height, top_left, year):
        self.url_template = url_template
        self.gsd = float(gsd)
        self.scale = float(scale)
        self.tile_width = float(tile_width)
        self.tile_height = float(tile_height)
        self.top_left = (float(top_left.split()[0]), float(top_left.split()[1]))
        self.year = int(year)
        self.pix2m = 0.00028  # According to OGC WMTS 1.0 spec for meters

    def coord2tile(self, cx, cy, int_cast=True):
        """
        Convert coordinates (cx, cy) to tile row and column at the given zoom level.
        """
        scale = self.scale
        tile_width_m = self.tile_width * scale * self.pix2m
        tile_height_m = self.tile_height * scale * self.pix2m
        top_left_x, top_left_y = self.top_left
        
        tx = float((cx - top_left_x) / tile_width_m)
        ty = float((top_left_y - cy) / tile_height_m)
        if int_cast == False:
            return tx, ty
        else:
            return int(tx), int(ty)

    def get_tile(self, cx, cy, tiles=False):
        """
        Fetch the image tile corresponding to the given coordinates (cx, cy).
        """
        if tiles == False:
            row, col = self.coord2tile(cx, cy)
        else:
            row, col = cx, cy
        url = self.url_template.format(TileRow=row, TileCol=col)
        response = requests.get(url)
        if response.ok:
            scale_down = int((self.tile_width * self.scale * self.pix2m)/self.gsd)
            image = Image.open(BytesIO(response.content))
            image = image.resize((scale_down, scale_down), resample=Image.NEAREST)
            return image
        else:
            raise Exception(f"Failed to fetch tile at {url}")
        
    def get_image(self, c1, c2):
        """
        Fetch tiles between two coordinates
        """
        x1, y1 = c1
        x2, y2 = c2
        x1, x2 = max(x1, x2), min(x1, x2)
        y1, y2 = max(y1, y2), min(y1, y2)

        xt2, yt1 = self.coord2tile(x1, y1)
        xt1, yt2 = self.coord2tile(x2, y2)

        x_size = xt2 - xt1 + 1
        y_size = yt2 - yt1 + 1

        # make a double list of tiles, this was hard
        tiles = [[self.get_tile(x, y, tiles=True) for x in range(xt1, xt2 + 1)] for y in range(yt1, yt2 + 1)]

        im_tile_width, im_tile_height = tiles[0][0].size
        im_width = im_tile_width * x_size
        im_height = im_tile_height * y_size

        # precreate the new image to paste in the loop
        stitched_img = Image.new(mode="RGB", size=(im_width, im_height))

        for row_idx, row in enumerate(tiles):
            for col_idx, tile in enumerate(row):
                stitched_img.paste(tile, (col_idx * im_tile_width, row_idx * im_tile_height))

        # Convert to numpy array in rasterio (bands, rows, cols)
        img_array = np.array(stitched_img).transpose((2, 0, 1))

        # convert the tile width and pos back to m coordinates for correct georeferencing
        scale_mpp = self.scale * self.pix2m  
        tile_width_m = self.tile_width * scale_mpp
        tile_height_m = self.tile_height * scale_mpp
        global_top_left_x, global_top_left_y = self.top_left
        origin_x = global_top_left_x + xt1 * tile_width_m
        origin_y = global_top_left_y - yt1 * tile_height_m

        transform = rasterio.transform.from_origin(origin_x, origin_y, self.gsd, self.gsd)

        # Profile for rasterio
        profile = {
            'driver': 'GTiff',
            'height': img_array.shape[1],
            'width': img_array.shape[2],
            'count': img_array.shape[0],
            'dtype': img_array.dtype,
            'crs': 'EPSG:28992',
            'transform': transform
        }
    
        with rasterio.open("output.tif", "w", **profile) as dst:
            dst.write(img_array)
        


class WMTSBuilder:
    def __init__(self, xml_url, identifier="EPSG:28992"):
        self.xml_url = xml_url
        self.identifier = identifier
        self.ns = {
            'wmts': 'http://www.opengis.net/wmts/1.0',
            'ows': 'http://www.opengis.net/ows/1.1'
        }
        self.url_template = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/{Layer}/{TileMatrixSet}/{TileMatrix}/{TileRow}/{TileCol}.jpeg"
        self.layer_info_list = []
        self.tilematrix = None
        self.scale = None
        self.top_left = None
        self.tile_width = None
        self.tile_height = None
        self._load_capabilities()

    def _load_capabilities(self):
        resp = requests.get(self.xml_url)
        if resp.ok:
            xml_str = resp.content
            root = etree.fromstring(xml_str)
            
            layers = root.findall("wmts:Contents/wmts:Layer", namespaces=self.ns)
            for layer in layers:
                layer_id_elem = layer.find("ows:Identifier", namespaces=self.ns)
                year_res = re.match(r"(\d\d\d\d)(?=_ortho(HR|25))", layer_id_elem.text)
                if year_res:
                    format_ = layer.find("wmts:Format", namespaces=self.ns)
                    year, res = year_res.groups()
                    if res == "HR":
                        res = 0.08
                    if res == "25":
                        res = 0.25
                    self.layer_info_list.append({
                        "layer": layer_id_elem.text,
                        "format": format_.text,
                        "tilematrixset": self.identifier,
                        "year": year,
                        "gsd": res
                    })
            
            # Parse tile matrix set (highest resolution)
            tile_matrix_sets = root.findall(f"wmts:Contents/wmts:TileMatrixSet/ows:Identifier[.='{self.identifier}']..", namespaces=self.ns)
            high_res_tilematrix = None
            for tms in tile_matrix_sets:
                scales = tms.findall("wmts:TileMatrix/wmts:ScaleDenominator", namespaces=self.ns)
                scale_values = [float(scale.text) for scale in scales]
                min_scale_idx = np.argmin(scale_values)
                high_res_tilematrix = scales[min_scale_idx].getparent()
            
            self.tilematrix = high_res_tilematrix.find("ows:Identifier", namespaces=self.ns).text
            self.scale = high_res_tilematrix.find("wmts:ScaleDenominator", namespaces=self.ns).text
            self.top_left = high_res_tilematrix.find("wmts:TopLeftCorner", namespaces=self.ns).text
            self.tile_width = high_res_tilematrix.find("wmts:TileWidth", namespaces=self.ns).text
            self.tile_height = high_res_tilematrix.find("wmts:TileHeight", namespaces=self.ns).text

    def build_maps(self):
        template_url_pre = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/{Layer}/{TileMatrixSet}/{TileMatrix}"
        template_url_suf = "/{TileRow}/{TileCol}.jpeg"
        maps = []
        for lyr in self.layer_info_list:
            url = template_url_pre.format(
                Layer=lyr["layer"],
                TileMatrixSet=lyr["tilematrixset"],
                TileMatrix=self.tilematrix
            )
            map_ = WMTSMap(url+template_url_suf, gsd=lyr["gsd"], scale=self.scale, tile_width=self.tile_width, tile_height=self.tile_height, top_left=self.top_left, year=lyr["year"])
            maps.append(map_)
        return maps