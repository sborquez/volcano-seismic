import os
import json
from pathlib import Path
import requests


DATASETS = dict()
BY_TAGS = dict()
def create_folder(output_folder):
    """Create dataset folder."""
    output_path = Path(output_folder)
    if output_path.exists():
        print(f"{output_path} already exists.")
    else:
        output_path.mkdir(parents=True, exist_ok=True)

def list_datasets(include_tags=[], return_list=False):
    """List datasets and their info."""
    global DATASETS
    global BY_TAGS
    filter_datasets = set(DATASETS.keys())
    if len(include_tags) > 0:
        filter_datasets = set()
        for tag in include_tags: filter_datasets.update(BY_TAGS[tag])
    print("Datasets:", len(filter_datasets))
    for k in filter_datasets:
        v = DATASETS[k]
        print(f"- {k}:\ntags: {list(v['tags'])}\n{v['info']}")
    if return_list: return list(filter_datasets)

def download_datasets(datasets=[], output=".", force=False):
    """Download Datasets."""
    global DATASETS
    global BY_TAGS
    print(f"Downloading {len(datasets)} datasets.")
    for i, dataset in enumerate(datasets):
        print(f"\t{i} - downloading '{dataset}' ...", end=" ")
        if force or not (Path(output)/dataset).exists():
            DATASETS[dataset]["download"](output)
            print("ok!")
        else:
            print("skipped")

def add_dataset(tags=[]):
    """Register a dataset."""
    def inner(func):
        global DATASETS
        global BY_TAGS 
        dataset_name = func.__name__
        DATASETS[dataset_name] = {
            "download" : func,
            "name": dataset_name,
            "info": func.__doc__,
            "tags":  set(tags)
        }
        for tag in tags:
            if tag not in BY_TAGS:
                BY_TAGS[tag] = []
            BY_TAGS[tag].append(dataset_name)
        return func
    return inner


@add_dataset(tags=["time-series", "classification"])
def llaima_volcano(output_folder):
    """Llaima volcano dataset: In-depth comparison of deep artificial neural network architectures on seismic events classification
    source: https://www.sciencedirect.com/science/article/pii/S2352340920305217
    """
    src = "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/dv8nwdd36k-1.zip"
    filename = "dv8nwdd36k-1.zip"
    output_file = Path(output_folder)/filename
    r = requests.get(src, allow_redirects=True)
    os.makedirs(Path(output_folder), exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(r.content)
    import zipfile
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(Path(output_folder))
    os.remove(output_file)
    return output_file

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Download Datasets.")
    ap.add_argument('-o', '--output', type=str,
                            help="Output folder.", default="/rapids/host/datasets")
    gp = ap.add_mutually_exclusive_group(required=True)
    gp.add_argument('-d', '--datasets', nargs="+", type=str)
    gp.add_argument('--ls', dest='list_datasets', action='store_true',
                            help="List available datasets.")
    gp.add_argument('--all', dest='download_all', action='store_true',
                            help="Download all datasets.")
    args = vars(ap.parse_args())
    
    output = args["output"]
    # List Datasets
    if args["list_datasets"]:
        list_datasets()
    # Download Datasets
    elif args["download_all"]:
        datasets = list(DATASETS.keys())
        download_datasets(datasets, output)
    elif len(args["datasets"]) > 0:
        datasets = args["datasets"]
        download_datasets(datasets, output)

