import hashlib
import os
import zipfile

import requests
from tqdm import tqdm


def download(url, path=None, overwrite=False) -> str:
    """Download an given URL.
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split("/")[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print("Downloading %s from %s..." % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        total_length = r.headers.get("content-length")
        with open(fname, "wb") as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=int(total_length / 1024.0 + 0.5),
                    unit="KB",
                    unit_scale=False,
                    dynamic_ncols=True,
                ):
                    f.write(chunk)
    return fname


def download_dataset(
    dataset_name: str, data_path: str = "data/", overwrite: bool = False
) -> None:
    compatible_datasets = [
        "ae_photos",
        "apple2orange",
        "cezanne2photo",
        "cityscapes",
        "facades",
        "grumpifycat",
        "horse2zebra",
        "iphone2dslr_flower",
        "maps",
        "mini",
        "mini_colorization",
        "mini_colorization",
        "mini_pix2pix",
        "monet2photo",
        "summer2winter_yosemi",
        "ukiyoe2photo",
        "vangogh2photo",
    ]
    if dataset_name not in compatible_datasets:
        print("The dataset you chose is not compatible.")
        print(f"Please select one among: {compatible_datasets}")
        return

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    download_url = f"https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{dataset_name}.zip"
    download_dir = os.path.join(data_path, "downloads")
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    filename = download(download_url, path=download_dir, overwrite=overwrite)

    # Extract archive in target dir
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(path=data_path)

    # Re-organize dirs for more clarity
    testdir = data_path + "test"
    traindir = data_path + "train"
    if not os.path.exists(testdir):
        os.mkdir(testdir)
    if not os.path.exists(traindir):
        os.mkdir(traindir)

    os.rename(data_path + "trainA", traindir + "/A")
    os.rename(data_path + "trainB", traindir + "/B")
    os.rename(data_path + "testA", testdir + "/A")
    os.rename(data_path + "testB", testdir + "/B")

    # Done
    print(f"Dataset downloaded and extracted in '{data_path}'.")
