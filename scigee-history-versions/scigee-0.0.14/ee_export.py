# coding=utf-8

""" 
https://github.com/giswqs/geemap/blob/master/AUTHORS.rst
Package: geemap
Author: Qiusheng Wu, qwu18@utk.edu
"""

import ee
import os
import requests
import shutil
import tarfile
import urllib.request
import zipfile

def ee_export_image(
    ee_object, filename, scale=None, crs=None, region=None, file_per_band=False
):
    """Exports an ee.Image as a GeoTIFF.
    Args:
        ee_object (object): The ee.Image to download.
        filename (str): Output filename for the exported image.
        scale (float, optional): A default scale to use for any bands that do not specify one; ignored if crs and crs_transform is specified. Defaults to None.
        crs (str, optional): A default CRS string to use for any bands that do not explicitly specify one. Defaults to None.
        region (object, optional): A polygon specifying a region to download; ignored if crs and crs_transform is specified. Defaults to None.
        file_per_band (bool, optional): Whether to produce a different GeoTIFF per band. Defaults to False.
    """

    if not isinstance(ee_object, ee.Image):
        print("The ee_object must be an ee.Image.")
        return

    filename = os.path.abspath(filename)
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    filetype = os.path.splitext(basename)[1][1:].lower()
    filename_zip = filename.replace(".tif", ".zip")

    if filetype != "tif":
        print("The filename must end with .tif")
        return

    try:
        print("Generating URL ...")
        params = {"name": name, "filePerBand": file_per_band}
        if scale is None:
            scale = ee_object.projection().nominalScale().multiply(10)
        params["scale"] = scale
        if region is None:
            region = ee_object.geometry()
        params["region"] = region
        if crs is not None:
            params["crs"] = crs

        try:
            url = ee_object.getDownloadURL(params)
        except Exception as e:
            print("An error occurred while downloading.")
            print(e)
            return
        print(f"Downloading data from {url}\nPlease wait ...")
        r = requests.get(url, stream=True)

        if r.status_code != 200:
            print("An error occurred while downloading.")
            return

        with open(filename_zip, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

    except Exception as e:
        print("An error occurred while downloading.")
        print(r.json()["error"]["message"])
        return

    try:
        with zipfile.ZipFile(filename_zip) as z:
            z.extractall(os.path.dirname(filename))
        os.remove(filename_zip)

        if file_per_band:
            print(f"Data downloaded to {os.path.dirname(filename)}")
        else:
            print(f"Data downloaded to {filename}")
    except Exception as e:
        print(e)

def ee_export_image_collection(
    ee_object, out_dir, scale=None, crs=None, region=None, file_per_band=False
):
    """Exports an ImageCollection as GeoTIFFs.
    Args:
        ee_object (object): The ee.Image to download.
        out_dir (str): The output directory for the exported images.
        scale (float, optional): A default scale to use for any bands that do not specify one; ignored if crs and crs_transform is specified. Defaults to None.
        crs (str, optional): A default CRS string to use for any bands that do not explicitly specify one. Defaults to None.
        region (object, optional): A polygon specifying a region to download; ignored if crs and crs_transform is specified. Defaults to None.
        file_per_band (bool, optional): Whether to produce a different GeoTIFF per band. Defaults to False.
    """

    if not isinstance(ee_object, ee.ImageCollection):
        print("The ee_object must be an ee.ImageCollection.")
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    try:

        count = int(ee_object.size().getInfo())
        print(f"Total number of images: {count}\n")

        for i in range(0, count):
            image = ee.Image(ee_object.toList(count).get(i))
            name = image.get("system:index").getInfo() + ".tif"
            filename = os.path.join(os.path.abspath(out_dir), name)
            print(f"Exporting {i + 1}/{count}: {name}")
            ee_export_image(
                image,
                filename=filename,
                scale=scale,
                crs=crs,
                region=region,
                file_per_band=file_per_band,
            )
            print("\n")

    except Exception as e:
        print(e)