from pystac_client import Client
import planetary_computer as pc
from odc.stac import load
from pyproj import Transformer 
import os


class SentinelMap:
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

        ds = load(
            signed_items,
            bands=all_bands,
            bbox=bounds,
            crs=self.crs,  
            resolution=10,
            chunks={},
            groupby="solar_day",
        ).sortby("time")

        return ds
    
    def get_sentinel_tile(self, tile_bounds, tile_name, cloud_cover=10, start_date="2015-01-01", end_date="2025-01-01"):

        ds = self.get_sentinel_data(bbox=tile_bounds, loud_cover=cloud_cover, start_date=start_date, end_date=end_date)
        # Save the xarray dataset to a NetCDF file
        output_file = os.path.join(self.dir, f"{tile_name}.nc")
        ds.to_netcdf(output_file, format='NETCDF4')
        print(f"Saved dataset to {output_file}")