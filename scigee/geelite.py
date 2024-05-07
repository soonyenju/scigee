import ee
import numpy as np
import pandas as pd
from datetime import datetime

def filter_collection(collection: str, date_range: list, bounds: ee.Geometry) -> str:
    """
    Get a list (collection) of ee images filtered by date range and ee geometry (point or so).
    Input:
    -------
    collection: str, ee collection name
    date_range: list, [start: str, end: str]
    bounds: ee.Geometry, e.g.
        ee.Geometry.Polygon(
        [[
        [minlon, maxlat],
        [maxlon, maxlat],
        [maxlon, minlat],
        [minlon, minlat]
        ]]
        ) 
    Return:
    -------
    A list of ee images (collection)
    """
    collection = ee.ImageCollection(collection)
    collection = collection.filterDate(*date_range)
    collection = collection.filterBounds(bounds)
    return collection.toList(collection.size())

def get_collection_size(collection: ee.ImageCollection) -> int:
    """
    Get the length of collection list
    Input:
    -------
    collecton: ee.ImageCollection
    Return:
    -------
    int, size of collection
    """
    return collection.size().getInfo()

def get_image(collection: ee.ImageCollection, count) -> ee.Image:
    """
    Get the ith image from collection
    Input:
    -------
    collection: ee.ImageCollection, a list of images
    count: index of wanted image in collection
    Return:
    -------
    ee.Image
    """
    image = collection.get(count)
    image = ee.Image(image)
    return image

def get_image_bandnames(image: ee.Image) -> list:
    """
    Get the band names of image
    Input:
    -------
    image: ee.Image
    Return:
    -------
    list, a list of band names
    """
    return image.bandNames().getInfo()

def get_proj(image: ee.Image, band: str, pyfmt: bool):
    """
    Get the projection information of an image
    Input:
    -------
    image: ee.Image
    band: str, band name
    pyfmt: bool, python or ee format of output projection
    Return:
    -------
    Projection of image
    """
    proj = image.select(band).projection()
    if pyfmt:
        proj = proj.getInfo()
    return proj
 
def proj_epsg(epsg: int or str = 4326) -> ee.Projection:
    proj = ee.Projection(f'EPSG:{epsg}')
    return proj
 
def get_scale(image: ee.Image) -> float:
    scale = image.projection().nominalScale().getInfo()
    return scale
 
def get_date(image):
    date = ee.Date(image.get('system:time_start'))
    return date.format('Y-M-d HH:mm:ss.SSSS').getInfo()

# # deprecated:
# def gee2df(collection_name, lat, lon, date_range, bandname, scale, radius = 0.001):
#     # unit degree, format:
#     # [minlon, minlat,
#     #  maxlon, maxlat]
#     roi = ee.Geometry.Rectangle([
#         lon - radius, lat - radius, 
#         lon + radius, lat+ radius
#     ])
#     start_date, end_date = date_range

#     collection = ee.ImageCollection(collection_name)\
#         .filterBounds(roi).filterDate(start_date, end_date)
#     # Sort the filtered collection by date in ascending order
#     collection = collection.sort('system:time_start')

#     def interp_image(image, bandname, scale):
#         image = ee.Image(image)
#         date = image.get('system:time_start')

#         image_band = image.select(bandname)
#         stats = image_band.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=scale)
#         val = stats.get(bandname)
#         return image.set('Info', [date, val])

#     # Map the function to the image collection
#     # collection = collection.map(interp_image)
#     collection = collection.map(lambda image: interp_image(image, bandname, scale))

#     # Use aggregate_array to get the values as an array
#     array = collection.aggregate_array('Info')


#     # Convert the array to a list using getInfo()
#     array = array.getInfo()
#     df = pd.DataFrame(array, columns = ['DATETIME', 'VALUE'])
#     df['DATETIME'] = df['DATETIME'].map(
#         lambda x: datetime.utcfromtimestamp(int(x) // 1000)
#     )
#     df = df.set_index('DATETIME')
#     return df

def gee2df(collection_name, lat, lon, date_range, bandnames, scale, radius = None):
    # unit degree, format:
    # [minlon, minlat,
    #  maxlon, maxlat]
    # radius unit deg, scale unit m
    if not radius:
        radius = scale / 1e5 * 2
    roi = ee.Geometry.Rectangle([
        lon - radius, lat - radius,
        lon + radius, lat+ radius
    ])
    start_date, end_date = date_range

    collection = ee.ImageCollection(collection_name)\
        .filterBounds(roi).filterDate(start_date, end_date)
    # Sort the filtered collection by date in ascending order
    collection = collection.sort('system:time_start')

    def interp_image(image, bandnames, scale):
        image = ee.Image(image)
        date = image.get('system:time_start')

        image_bands = image.select(bandnames)
        stats = image_bands.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=scale)
        # val = stats.get(bandname)
        val_list = [stats.get(bandname) for bandname in bandnames]
        return image.set('Info', [date] + val_list)

    # Map the function to the image collection
    # collection = collection.map(interp_image)
    collection = collection.map(lambda image: interp_image(image, bandnames, scale))

    # Use aggregate_array to get the values as an array
    array = collection.aggregate_array('Info')


    # Convert the array to a list using getInfo()
    array = array.getInfo()
    df = pd.DataFrame(array, columns = ['DATETIME'] + bandnames)
    df['DATETIME'] = df['DATETIME'].map(
        lambda x: datetime.utcfromtimestamp(int(x) // 1000)
    )
    df = df.set_index('DATETIME')
    return df

def gee2drive(image, roi, name, folder, scale):
    if not type(roi) == ee.geometry.Geometry:
        minlon = roi[0]
        maxlon = roi[2]
        minlat = roi[1]
        maxlat = roi[3]
        roi = ee.Geometry.BBox(minlon, minlat, maxlon, maxlat)

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=folder,
        region=roi,
        scale=scale,
        maxPixels = 1e13,
        crs='EPSG:4326'
    )
    task.start()

def gee2local(ee_object, filename, scale, roi, user_params = {}, timeout = 300, proxies = None):
    '''
    Code is from geemap (https://github.com/gee-community/geemap)
    '''
    import os, requests, zipfile, pathlib
    if type(filename) == pathlib.PosixPath:
        filename = filename.as_posix()
    filename_zip = filename.replace(".tif", ".zip")

    params = {
        "dimensions": None,
        "crs": None,
        "crs_transform": None,
        "format": "ZIPPED_GEO_TIFF"

    }
    params["scale"] = scale
    params["region"] = roi
    params.update(user_params)

    try:
        url = ee_object.getDownloadURL(params)
        r = requests.get(url, stream = True, timeout = timeout, proxies = proxies)

        if r.status_code != 200:
            print("An error occurred while downloading.")
            return

        with open(filename_zip, "wb") as fd:
            for chunk in r.iter_content(chunk_size = 1024):
                fd.write(chunk)

    except Exception as e:
        print("An error occurred while downloading.")
        if r is not None:
            print(r.json()["error"]["message"])
        return

    try:
        with zipfile.ZipFile(filename_zip) as z:
            z.extractall(os.path.dirname(filename))
        os.remove(filename_zip)

    except Exception as e:
        print(e)