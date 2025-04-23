import requests
import os
from tqdm import tqdm
import re

class TileDownloader:
    def __init__(self, url_template, output_dir):
        """
        Initialize the downloader with a URL template and output directory.
        """
        self.url_template = url_template
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download_tile(self, **kwargs):
        """
        Download a tile using any number of formatting parameters for the URL template.
        The downloaded file will be saved with a filename based on the kwargs.
        """
        # make dir if not exist
        if not self.output_dir:
            raise ValueError("Output directory is not set.")

        file_name = "_".join(str(v) for v in kwargs.values()) + ".tif"
        file_path = os.path.join(self.output_dir, file_name)

        # quick check to see if the file is already there
        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping download.")
            return file_path

        url = self.url_template.format(**kwargs)
        response = requests.get(url, stream=True)
        if response.ok:
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded tile to {file_path}")
            return file_path
        else:
            raise Exception(f"Failed to fetch tile at {url} (status code: {response.status_code})")

def download_url(url, directory, filename = None, chunk_size = 1048576):
    """
    Downloads the file from a url to a filepaht in chucks of 1MB with some error handeling
    """
    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()  # Raise error for bad status if applicable
            total = int(response.headers.get("content-length", 0))
            if filename is None:
                filename = re.split(r"https?:\/\/.*\/(.*\.[^.]+$)", url)[1] # regex to find the filename at the end, see capture group
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