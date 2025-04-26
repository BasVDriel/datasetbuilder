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

        file_path = os.path.join(self.output_dir, file_name+ original_extension)

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

    try:
        with requests.get(url, stream=True, timeout=30, headers=headers) as response:
            response.raise_for_status()  # Raise error for bad status if applicable
            total = int(response.headers.get("content-length", 0))
            filepath = os.path.join(directory, filename)

            with open(filepath, "wb") as file, tqdm(desc=filepath, total=total, unit="B", unit_scale=True) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
    
def delete_recursively( path, test=False):
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