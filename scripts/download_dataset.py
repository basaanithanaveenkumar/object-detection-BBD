import os
import requests
import zipfile
from tqdm import tqdm

# BDD100K dataset download links
DATASET_URLS = {
    #"images_100k": "http://128.32.162.150/bdd100k/bdd100k_images_100k.zip",
    "labels": "http://128.32.162.150/bdd100k/bdd100k_labels.zip",
}


def download_file(url, save_dir="bdd100k"):
    """
    Download a file from URL with progress bar
    """
    os.makedirs(save_dir, exist_ok=True)
    local_filename = os.path.join(save_dir, url.split("/")[-1])

    # Streaming download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(local_filename, "wb") as f, tqdm(
            desc=local_filename,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
    return local_filename


def unzip_file(zip_path, extract_to=None):
    """
    Unzip a downloaded file
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def download_bdd100k(components=None, save_dir="bdd100k"):
    """
    Download specified components of BDD100K dataset

    Args:
        components (list): List of components to download ('images_100k', 'labels',
                          'drivable_maps', 'segmentation'). If None, downloads all.
        save_dir (str): Directory to save the dataset
    """
    if components is None:
        components = DATASET_URLS.keys()

    for component in components:
        if component not in DATASET_URLS:
            print(f"Warning: Unknown component '{component}', skipping")
            continue

        print(f"\nDownloading {component}...")
        url = DATASET_URLS[component]
        zip_path = download_file(url, save_dir)
        unzip_file(zip_path)
        os.remove(zip_path)  # Remove the zip file after extraction


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download BDD100K dataset")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=list(DATASET_URLS.keys()),
        help="Components to download (default: all)",
    )
    parser.add_argument(
        "--save_dir", default="data/", help="Directory to save the dataset"
    )

    args = parser.parse_args()

    download_bdd100k(components=args.components, save_dir=args.save_dir)
    print("\nBDD100K dataset download complete!")
