import requests

class ElevationModel:
    def __init__(self, elevation_model):
        elevation_model = elevation_model.lower()
        if elevation_model == "dem" or elevation_model == "dtm":
            self.em = elevation_model
        else:
             raise ValueError("Only DEM and DTM are allowed arguments")
        self.url_template = "https://service.pdok.nl/rws/ahn/atom/downloads/{em}_05m/{tile}.tif"

    def download_tile(self, tile_str):
        url = self.url_template.format(em=self.em, tile=tile_str)
        response = requests.get(url)
        if response.ok:
            pass
        else:
            raise Exception(f"Failed to fetch tile at {url}")