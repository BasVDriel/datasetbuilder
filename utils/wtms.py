import requests
import re
import rasterio 
from lxml import etree
import numpy as np
from io import BytesIO
from PIL import Image
import cv2


class WMTSMap:
    """
    Represents a WMTS map source. Fetches image tiles and assembles full images based on coordinates.
    
    Inputs:
        url_template (str), gsd (float), scale (float), tile size (int), top_left (str), year (int)
    Outputs:
        Initialized WMTSMap object
    """
    def __init__(self, url_template, gsd, scale, tile_width, tile_height, top_left, year):
        self.url_template = url_template
        self.gsd = float(gsd)
        self.scale = float(scale)
        self.tile_width = float(tile_width)
        self.tile_height = float(tile_height)
        self.top_left = (float(top_left.split()[0]), float(top_left.split()[1]))
        self.year = int(year)
        self.pix2m = 0.00028  # According to OGC WMTS 1.0 spec for meters
        self.scaledown = int((self.tile_width * self.scale * self.pix2m)/self.gsd)

    def coord2tile(self, cx, cy, int_cast=True):
        """
        Converts EPSG:28992 coordinates to tile indices.

        Inputs: cx, cy (float), int_cast (bool), these are x and y parts of coordinates
        Outputs: tile indices (int or float)
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
        Downloads a tile image from given coordinates or tile indices (no resize).

        Inputs: cx, cy (float or int), tiles (bool)
        Outputs: PIL.Image tile
        """
        if not tiles:
            row, col = self.coord2tile(cx, cy)
        else:
            row, col = cx, cy
        url = self.url_template.format(TileRow=row, TileCol=col)
        response = requests.get(url)
        if response.ok:
            image = Image.open(BytesIO(response.content))
            return image  # No resize here
        else:
            raise Exception(f"Failed to fetch tile at {url}")
        
    def get_image(self, c1, c2):
        """
        Fetches and stitches all tiles between two coordinates.

        Inputs: c1, c2 (tuple of float) coordinates
        Outputs: PIL.Image of combined area
        """
        x1, y1 = c1
        x2, y2 = c2
        x1, x2 = max(x1, x2), min(x1, x2)
        y1, y2 = max(y1, y2), min(y1, y2)

        xt2, yt1 = self.coord2tile(x1, y1)
        xt1, yt2 = self.coord2tile(x2, y2)

        x_size = xt2 - xt1 + 1
        y_size = yt2 - yt1 + 1

        tiles = [[self.get_tile(x, y, tiles=True) for x in range(xt1, xt2 + 1)] for y in range(yt1, yt2 + 1)]

        im_tile_width, im_tile_height = tiles[0][0].size
        im_width = im_tile_width * x_size
        im_height = im_tile_height * y_size

        stitched_img = Image.new(mode="RGB", size=(im_width, im_height))

        for row_idx, row in enumerate(tiles):
            for col_idx, tile in enumerate(row):
                stitched_img.paste(tile, (col_idx * im_tile_width, row_idx * im_tile_height))

        return stitched_img
        
    def crop_to_bbox(self, stitched_img, top_left, bottom_right):
        """
        Crops the stitched image exactly to bounding box (in original resolution).

        Inputs:
            stitched_img (PIL.Image)
            top_left (tuple) - EPSG:28992 (x, y)
            bottom_right (tuple) - EPSG:28992 (x, y)

        Returns:
            Cropped PIL.Image, pixel offsets (for geotransform adjustment)
        """
        xtl_float, ytl_float = self.coord2tile(*top_left, int_cast=False)
        xbr_float, ybr_float = self.coord2tile(*bottom_right, int_cast=False)
        xtl_int, ytl_int = self.coord2tile(*top_left, int_cast=True)
        xbr_int, ybr_int = self.coord2tile(*bottom_right, int_cast=True)

        tile_size_pixels_x = stitched_img.size[0] / (xbr_int - xtl_int + 1)
        tile_size_pixels_y = stitched_img.size[1] / (ybr_int - ytl_int + 1)

        left = (xtl_float - xtl_int) * tile_size_pixels_x
        top = (ytl_float - ytl_int) * tile_size_pixels_y
        right = (xbr_float - xtl_int) * tile_size_pixels_x
        bottom = (ybr_float - ytl_int) * tile_size_pixels_y

        cropped_img = stitched_img.crop((
            int(left),
            int(top),
            int(right),
            int(bottom)
        ))

        # return pixel offsets to update transform
        return cropped_img, (left, top)


    def save_image(self, img, top_left_coord, bottom_right_coord, path="output.tif"):
        """
        Crops the image to the bounding box and saves as a GeoTIFF with correct geotransform.

        Inputs:
            img (PIL.Image): The stitched image (PIL format).
            top_left_coord (tuple): EPSG:28992 coordinate for the top-left corner (x, y).
            bottom_right_coord (tuple): EPSG:28992 coordinate for the bottom-right corner (x, y).
            path (str): The path to save the resulting GeoTIFF file.
        """
        # Crop the image to the bounding box
        cropped_img, pixel_offset = self.crop_to_bbox(img, top_left_coord, bottom_right_coord)

        # Calculate the adjusted geotransform
        scale_mpp = self.scale * self.pix2m
        tile_width_m = self.tile_width * scale_mpp
        tile_height_m = self.tile_height * scale_mpp
        global_top_left_x, global_top_left_y = self.top_left

        # Use the corrected coord2tile method to get float values
        xtl_float, ytl_float = self.coord2tile(*top_left_coord, int_cast=False)

        # Origin calculation based on the bounding box and pixel offsets
        origin_x = global_top_left_x + xtl_float * tile_width_m 
        origin_y = global_top_left_y - ytl_float * tile_height_m 

        # Create the geotransform for the cropped image
        transform = rasterio.transform.from_origin(origin_x, origin_y, self.gsd, self.gsd)

        # resize image to the true size (bounding box size) devided by gsd
        x_size = int((bottom_right_coord[0] - top_left_coord[0]) / self.gsd)
        y_size = int((top_left_coord[1] - bottom_right_coord[1]) / self.gsd)

        resized_img = cropped_img.resize((x_size, y_size), resample=Image.NEAREST)
        resized_img = np.array(resized_img).transpose((2, 0, 1))  # (bands, y, x)

        profile = {
            'driver': 'GTiff',
            'count': resized_img.shape[0],
            'dtype': 'uint8',
            'width': resized_img.shape[2],
            'height': resized_img.shape[1],
            'crs': 'EPSG:28992',
            'transform': transform
        }

        # Save the cropped image as a GeoTIFF
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(resized_img)



class WMTSBuilder:
    """
    Builds WMTSMap instances by parsing WMTS capabilities XML.

    Inputs: xml_url (str), identifier (str), detail (int)
    Outputs: Initialized WMTSBuilder with parsed layer data
    """
    def __init__(self, xml_url, identifier="EPSG:28992", detail=19):
        self.detail = detail
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
        self.load_capabilities()

    def load_capabilities(self):
        """
        Parses XML capabilities and extracts layer and tile matrix info.

        Inputs: None
        Outputs: Sets internal state with WMTS metadata
        """
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
            
            tile_matrix_sets = root.findall(f"wmts:Contents/wmts:TileMatrixSet/ows:Identifier[.='{self.identifier}']..", namespaces=self.ns)
            high_res_tilematrix = None
            for tms in tile_matrix_sets:
                scales = tms.findall("wmts:TileMatrix/wmts:ScaleDenominator", namespaces=self.ns)
                min_scale_idx = self.detail
                high_res_tilematrix = scales[min_scale_idx].getparent()
            
            self.tilematrix = high_res_tilematrix.find("ows:Identifier", namespaces=self.ns).text
            self.scale = high_res_tilematrix.find("wmts:ScaleDenominator", namespaces=self.ns).text
            self.top_left = high_res_tilematrix.find("wmts:TopLeftCorner", namespaces=self.ns).text
            self.tile_width = high_res_tilematrix.find("wmts:TileWidth", namespaces=self.ns).text
            self.tile_height = high_res_tilematrix.find("wmts:TileHeight", namespaces=self.ns).text

    def build_maps(self):
        """
        Constructs WMTSMap objects for all parsed layers.

        Inputs: None
        Outputs: List of WMTSMap objects
        """
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
