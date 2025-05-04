from pystac_client import Client
import planetary_computer as pc
from odc.stac import load
from pyproj import Transformer 
import warnings
from rasterio.errors import NotGeoreferencedWarning
import os


class SentinelDownloader:
    def __init__(self, url, sentinel_dir, crs="EPSG:28992"):
        self.url = url
        self.catalog = Client.open(self.url)
        self.crs = crs
        self.dir = sentinel_dir

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
    
    def download_tile(self, tile_bounds, tile_name, cloud_cover=10, start_date="2015-01-01", end_date="2025-01-01"):
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        if not self.dir:
            raise ValueError("Output directory is not set.")
        
        output_path = os.path.join(self.dir, f"{tile_name}.nc")
        if os.path.exists(output_path):
            print(f"File {output_path} already exists extracted. Skipping download.")
            return output_path

        print(f"Downloading {output_path}")
        ds = self.get_sentinel_data(bbox=tile_bounds, cloud_cover=cloud_cover, start_date=start_date, end_date=end_date)
        ds.to_netcdf(output_path, format='NETCDF4')

        return output_path