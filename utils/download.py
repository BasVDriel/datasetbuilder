import logging
import os
import re
import time
import csv
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import planetary_computer as pc
import requests
from lxml import etree
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm
from odc.stac import load
from pystac_client import Client
from pyproj import Transformer


class Downloader:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        output_dir_csv = os.path.basename(f"{self.output_dir}.csv")
        self.tracker_path = os.path.join(self.output_dir, output_dir_csv)
        self.file_inventory()

    def file_inventory(self):
        """
        Reads the CSV file and stores the file information in a DataFrame.
        """
        if not os.path.exists(self.tracker_path):
            self.file_data = pd.DataFrame(columns=['file_path', 'status'])
            self.file_data.to_csv(self.tracker_path, index=False)
        else:
            self.file_data = pd.read_csv(self.tracker_path)

    def file_exists(self, filepath):
        """
        Checks if the given file path exists in the DataFrame.
        """
        return filepath in self.file_data['file_path'].values
    
    @staticmethod
    def download_wrapper(func):
        def wrapped(*args, **kwargs):
            try:
                path = func(*args, **kwargs)
                status = True
            except Exception as e:
                print(e)
                status = False
            
            args[0].update_file_tracker(path, status)
            return path
        return wrapped

    def update_file_tracker(self, file_path, status):
        if self.file_exists(file_path):
            self.file_data.loc[self.file_data['file_path'] == file_path, ['status']] = [status]
        else:
            new_row = {'file_path': file_path, 'status': status}
            self.file_data = pd.concat([self.file_data, pd.DataFrame([new_row])], ignore_index=True)

        self.file_data.to_csv(self.tracker_path, index=False)

    def unzip(self, file_path, file_name):
        """
        Unzip a file to the output directory, rename it with a filename and delete the original zip file.
        """
        return_path = None
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)
            for file_ref in zip_ref.namelist():
                original_extension = os.path.splitext(file_ref)[1]
                new_file_path = os.path.join(self.output_dir, file_name + original_extension)
                return_path = new_file_path
                os.rename(os.path.join(self.output_dir, file_ref), new_file_path)  # rename the extracted file
        # remove the original zip file
        os.remove(file_path)
        return return_path








class SentinelDownloader:
    def __init__(self, url, output_dir, crs="EPSG:28992"):
        self.url = url
        self.catalog = Client.open(self.url)
        self.crs = crs
        self.output_dir = output_dir

    def get_sentinel_data(self, bbox, cloud_cover=10, start_date="2015-01-01", end_date="2025-01-01"):
        time_of_interest = f"{start_date}/{end_date}"

        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        min_lon, min_lat = transformer.transform(bbox[0], bbox[1])
        max_lon, max_lat = transformer.transform(bbox[2], bbox[3])
        bounds = (min_lon, min_lat, max_lon, max_lat)

        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bounds,
            query={"eo:cloud_cover": {"lt": cloud_cover}},
            datetime=time_of_interest,
        )
    
        signed_items = [pc.sign(item) for item in search.items()]

        # Load bands
        all_bands = [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
            "B09", "B11", "B12", "SCL", "AOT", "WVP"
        ]

        ds = load(  # your existing load() call
            signed_items,
            bands=all_bands,
            bbox=bounds,
            crs=self.crs,
            resolution=10,
            chunks={},
            groupby="solar_day",
        ).sortby("time")

        return ds
    
    @Downloader.download_wrapper
    def download_tile(self, tile_bounds, tile_name, cloud_cover=10, start_date="2015-01-01", end_date="2025-01-01"):
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        if not self.output_dir:
            raise ValueError("Output directory is not set.")
        
        output_path = os.path.join(self.output_dir, f"{tile_name}.nc")
        if os.path.exists(output_path):
            print(f"File {output_path} already exists extracted. Skipping download.")
        else:
            print(f"Downloading {output_path}")
            ds = self.get_sentinel_data(bbox=tile_bounds, cloud_cover=cloud_cover, start_date=start_date, end_date=end_date)
            ds.to_netcdf(output_path, format='NETCDF4')

        return output_path
    




class TileDownloader(Downloader):
    def __init__(self, url_template, output_dir):
        """
        Initialize the downloader with a URL template and output directory.
        """
        super().__init__(output_dir)
        self.url_template = url_template

    @Downloader.download_wrapper      
    def download_tile(self, file_name, unzip=False, **kwargs):
        """
        Download a tile using any number of formatting parameters for the URL template.
        The downloaded file will be saved with a filename based on the kwargs.
        """
        # make dir if not exist
        if not self.output_dir:
            raise ValueError("Output directory is not set.")

        url = self.url_template.format(**kwargs)
        match = re.search(r"https?:\/\/.*\/(.*\.[^.]+$)", url)
        if not match:
            raise ValueError(f"Could not extract filename from URL: {url}")
        original_extension = "." + match.group(1).split(".")[-1]

        file_path = os.path.join(self.output_dir, file_name+original_extension)

        # check for alternative paths
        alternative_extensions = ['.tif', '.laz', ".tiff", ".geojson"]
        alternative_extensions += [ext.upper() for ext in alternative_extensions]
        for ext in alternative_extensions:
            alternative_path_dir = os.path.dirname(file_path)
            alternative_path = os.path.join(alternative_path_dir, file_name+ext)

            if os.path.exists(alternative_path):
                print(f"File {alternative_path} already exists extracted. Skipping download.")
                return alternative_path

        file_path = download_url(url, self.output_dir, filename=file_name + original_extension)

        if unzip == True and file_path.endswith(".zip"):
            file_path = self.unzip(file_path, file_name)

        return file_path
            

def download_url(url, directory, filename = None, chunk_size = 1048576):
    """
    Downloads the file from a url to a filepaht in chucks of 1MB with some error handeling
    """
    # alternative user agent from wget to spoof the
    headers = {
        "User-Agent": "Wget/1.21.2",
        "Accept": "*/*",
        "Accept-Encoding": "identity",
        "Connection": "Keep-Alive"
    }
    with requests.get(url, stream=True, timeout=30, headers=headers) as response:
        response.raise_for_status()  # Raise error for bad status if applicable
        total = int(response.headers.get("content-length", 0))
        filepath = os.path.join(directory, filename)

        with open(filepath, "wb") as file, tqdm(desc=filepath, total=total, unit="B", unit_scale=True) as bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
    return filepath

    
def delete_recursively(path, test=False):
    """
    Deletes a directory and only its contents
    """
    subdirs = [f.path for f in os.scandir(path) if f.is_dir()]
    files = [f.path for f in os.scandir(path) if f.is_file()]
    for subdir in subdirs:
        delete_recursively(subdir, test=test)
    for file in files:
        if test:
            print(f"Would delete: {file}")
        else:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
    return 








class WMTSMap(Downloader):
    """
    Represents a WMTS map source. Fetches image tiles and assembles full images based on coordinates.
    
    Inputs:
        url_template (str), gsd (float), scale (float), tile size (int), top_left (str), year (int)
    Outputs:
        Initialized WMTSMap object
    """
    def __init__(self, output_dir, url_template, gsd, scale, tile_width, tile_height, top_left, year):
        super().__init__(output_dir)
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

    def get_image(self, c1, c2, timing=False):
        """
        Fetches and stitches all tiles between two coordinates using multithreading.

        Inputs:
            c1, c2 (tuple of float) coordinates
        Outputs:
            PIL.Image of combined area
        """
        if timing:
            t1 = time.time()

        x1, y1 = c1
        x2, y2 = c2
        x1, x2 = max(x1, x2), min(x1, x2)
        y1, y2 = max(y1, y2), min(y1, y2)

        xt2, yt1 = self.coord2tile(x1, y1)
        xt1, yt2 = self.coord2tile(x2, y2)

        x_size = xt2 - xt1 + 1
        y_size = yt2 - yt1 + 1

        coords = [(x, y) for y in range(yt1, yt2 + 1) for x in range(xt1, xt2 + 1)]

        # Use threads for I/O-bound tile downloading
        with ThreadPoolExecutor(max_workers=10) as executor:
            tiles_flat = list(executor.map(lambda xy: self.get_tile(*xy, tiles=True), coords))

        # Convert flat tile list back to 2D grid (rows of tiles)
        tiles = [tiles_flat[i * x_size:(i + 1) * x_size] for i in range(y_size)]

        im_tile_width, im_tile_height = tiles[0][0].size
        im_width = im_tile_width * x_size
        im_height = im_tile_height * y_size

        stitched_img = Image.new(mode="RGB", size=(im_width, im_height))

        for row_idx, row in enumerate(tiles):
            for col_idx, tile in enumerate(row):
                stitched_img.paste(tile, (col_idx * im_tile_width, row_idx * im_tile_height))

        if timing:
            t2 = time.time()
            tdiff = t2-t1
        else:
            tdiff = 0

        return stitched_img, tdiff

        
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

    @Downloader.download_wrapper
    def save_image(self, top_left_coord, bottom_right_coord, file_name="output.tif", img=False):
        """
        Crops the image to the bounding box and saves as a GeoTIFF with correct geotransform.

        Inputs:
            img (PIL.Image): The stitched image (PIL format).
            top_left_coord (tuple): EPSG:28992 coordinate for the top-left corner (x, y).
            bottom_right_coord (tuple): EPSG:28992 coordinate for the bottom-right corner (x, y).
            file_name (str): The file_name to save the resulting GeoTIFF file.
        """
        if img == False:
            img, timing = self.get_image(top_left_coord, bottom_right_coord)

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
        image_path = os.path.join(self.output_dir, file_name)
        with rasterio.open(image_path, "w", **profile) as dst:
            dst.write(resized_img)

        return image_path


class WMTSBuilder:
    """
    Builds WMTSMap instances by parsing WMTS capabilities XML.

    Inputs: xml_url (str), identifier (str), detail (int)
    Outputs: Initialized WMTSBuilder with parsed layer data
    """
    def __init__(self, source_dir, xml_url, identifier="EPSG:28992", detail=19):
        self.source_dir = source_dir
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
            output_dir = os.path.join(self.source_dir, lyr["layer"])
            map_ = WMTSMap(output_dir, url+template_url_suf, gsd=lyr["gsd"], scale=self.scale, tile_width=self.tile_width, tile_height=self.tile_height, top_left=self.top_left, year=lyr["year"])
            maps.append(map_)
        return maps
