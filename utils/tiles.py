import requests
import os
import re
from tqdm import tqdm
from zipfile import ZipFile



class TileDownloader:
    def __init__(self, url_template, output_dir):
        """
        Initialize the downloader with a URL template and output directory.
        """
        self.url_template = url_template
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download_tile(self, file_name, unzip=False, **kwargs):
        """
        Download a tile using any number of formatting parameters for the URL template.
        The downloaded file will be saved with a filename based on the kwargs.
        """
        # make dir if not exist
        if not self.output_dir:
            raise ValueError("Output directory is not set.")

        url = self.url_template.format(**kwargs)
        file_path = os.path.join(self.output_dir, file_name)

        # check for alternative paths
        alternative_extensions = ['.tif', '.laz', ".tiff", ".geojson", ".zip"]
        alternative_extensions += [ext.upper() for ext in alternative_extensions]
        for ext in alternative_extensions:
            alternative_path_dir = os.path.dirname(file_path)
            alternative_path = os.path.join(alternative_path_dir, file_name+ext)

            if os.path.exists(alternative_path):
                print(f"File {alternative_path} already exists extracted. Skipping download.")
                return alternative_path

        download_url(url, self.output_dir, filename=file_name)
        if unzip == True:
            # loading the temp.zip and creating a zip object 
            with ZipFile(file_path, 'r') as z_object: 
                z_object.extractall(self.output_dir)
                z_object.close() 
            os.remove(file_path)

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

    try:
        with requests.get(url, stream=True, timeout=30, headers=headers) as response:
            response.raise_for_status()  # Raise error for bad status if applicable
            total = int(response.headers.get("content-length", 0))
            filepath = os.path.join(directory, filename)

            with open(filepath, "wb") as file, tqdm(desc=filename, total=total, unit="B", unit_scale=True) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)
        print(f"Downloaded to {filepath}")
        return filename

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None