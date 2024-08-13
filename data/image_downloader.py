import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "utils"))
)

import argparse
import concurrent.futures
import csv
import time
from pathlib import Path

import requests
from utils.logger import logger

IMG_DIR = "./data/images"

def download_image(url):
    """Download image from url and save it to filename"""
    filename = url.split("/")[-1]
    file = Path(IMG_DIR).joinpath(filename)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("wb") as handle:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", help="CSV file with image URLs")
    args = parser.parse_args()

    with open(args.csv_file, "r") as handle:
        reader = csv.reader(handle)
        urls = [r[0] for i, r in enumerate(reader) if i > 0]

    t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_image, urls)

    logger.info(f"Downloaded {len(urls)} images in {time.perf_counter() - t:.2f} seconds")
