import ee
import numpy as np
import pandas as pd
from datetime import datetime

def init_gee(project_name):
    import ee

    # Trigger the authentication flow.
    ee.Authenticate()

    # Initialize the library.
    ee.Initialize(project = project_name)

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

def collection2ts(collection, roi, date_range, bandnames, scale, radius = None):
    import geopandas as gpd
    # radius unit deg, scale unit m
    # EXAMPLE: collection2ts('MODIS/061/MCD15A3H', [52.51, -0.13], ['2020-01-01', '2020-03-01'], ['Lai'], 500, radius = None)
    def interp_image(image, bandnames, scale):
        image = ee.Image(image)
        date = image.get('system:time_start')

        image_bands = image.select(bandnames)
        stats = image_bands.reduceRegion(reducer = ee.Reducer.mean(), geometry = roi, scale=scale)
        # val = stats.get(bandname)
        val_list = [stats.get(bandname) for bandname in bandnames]
        return image.set('Info', [date] + val_list)
    
    if type(roi) == list:
        lat, lon = roi
        if not radius:
            radius = scale / 1e5 * 2
        # [minlon, minlat, maxlon, maxlat]
        roi = ee.Geometry.Rectangle([
            lon - radius, lat - radius,
            lon + radius, lat+ radius
        ])
    elif type(roi) == gpd.geodataframe.GeoDataFrame:
        import json
        geojson_dict = json.loads(roi.dissolve().to_json())
        roi = ee.FeatureCollection(geojson_dict["features"])
    else:
        assert type(roi) == ee.FeatureCollection, 'roi invalid.'
    start_date, end_date = date_range

    if type(collection) == str:
        collection = ee.ImageCollection(collection)
    else:
        assert type(collection) == ee.imagecollection.ImageCollection, 'collection invalid.'
    collection = collection.filterBounds(roi).filterDate(start_date, end_date)
    # Sort the filtered collection by date in ascending order
    collection = collection.sort('system:time_start')

    # Map the function to the image collection
    # collection = collection.map(interp_image)
    collection = collection.map(lambda image: interp_image(image.clip(roi), bandnames, scale))

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

def gee2df(collection, lat, lon, date_range, bandnames, scale, radius = None):
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

    if type(collection) == str:
        collection = ee.ImageCollection(collection)
    else:
        assert type(collection) == ee.imagecollection.ImageCollection, 'collection invalid.'
    collection = collection.filterBounds(roi).filterDate(start_date, end_date)
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

def image2points(image, dfi, scale, longitude = 'longitude', latitude = 'latitude', batch_size = 100):
    from tqdm import tqdm
    dfi = dfi.copy().reset_index()

    # Split DataFrame into batches using np.array_split (handles uneven splits)
    df_batches = np.array_split(dfi, np.ceil(len(dfi) / batch_size))

    # Initialize lists to store results
    results = []

    # Process each batch
    for batch in tqdm(df_batches, desc = 'iterating batches...'):
        if batch.empty: continue

        points = ee.FeatureCollection([
            ee.Feature(
                ee.Geometry.Point(lon_, lat_),
                {**{'longitude': lon_, 'latitude': lat_}, **row.to_dict()}
            )
            for (_, row), (lon_, lat_) in zip(batch.iterrows(), zip(batch[longitude], batch[latitude]))
        ])

        # Sample the image at batch points
        samples = image.sampleRegions(collection = points, scale = scale)

        # Extract values, and index (name of point) from the results
        for feature in samples.getInfo()['features']:
            results.append(pd.Series(feature['properties']).to_frame().T)

    df_results = pd.concat(results, axis = 0)
    return df_results

def gdf2ee(gdf):
    features = []

    for _, row in gdf.iterrows():
        # Convert geometry to GeoJSON and then to EE Geometry
        geom = ee.Geometry(row.geometry.__geo_interface__)

        # Convert attributes to a dictionary
        properties = row.drop('geometry').to_dict()

        # Create an EE Feature
        feature = ee.Feature(geom, properties)
        features.append(feature)

    return ee.FeatureCollection(features)

def image4shape(image, gdf, scale, to_xarray = False, properties = []):
    # Sample the image using the shapefile
    samples = image.sampleRegions(
        collection = gdf2ee(gdf),  # FeatureCollection
        scale = scale,  # Spatial resolution (meters)
        properties = properties,
        geometries = True  # Keep geometry for spatial analysis
    )
    values = samples.getInfo()

    # Extract elevation, slope, and index (name of point) from the results
    dfo = []
    for feature in values['features']:
        dfo.append(
            pd.concat([
                pd.Series(feature['geometry']['coordinates'], index = ['longitude', 'latitude']),
                pd.Series(feature['properties'])
            ]).to_frame().T
        )

    dfo = pd.concat(dfo, axis = 0)
    if to_xarray:
        dfo = dfo.sort_values(by = ['latitude', 'longitude']).set_index(['latitude', 'longitude']).to_xarray()
    return dfo


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

def get_status():
    tasks = ee.data.listOperations()
    pending_tasks = sum(1 for task in tasks if task['metadata']['state'] == 'PENDING')
    return pending_tasks

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

def interactive_map(dataset, vis_params, name, attr = 'Google Earth Engine', opacity = 0.5, base_map = 'terrain'):
    import folium

    # Ensure opacity is in visualization parameters
    vis_params["opacity"] = opacity

    # Get a GEE tile layer URL
    map_id_dict = dataset.getMapId(vis_params)
    tile_url = map_id_dict['tile_fetcher'].url_format

    # Create a folium map centered at some location
    m = folium.Map(location = [20, 0], zoom_start=2)

    if base_map == 'satellite':
        # Add satellite base layer (ESRI Satellite)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            overlay=False
        ).add_to(m)
    elif base_map == 'satellite, Google':
        # Add satellite base layer (Google Satellite)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            name="Satellite",
            overlay=False
        ).add_to(m)

    # Add GEE layer to folium map with transparency
    folium.raster_layers.TileLayer(
        tiles=tile_url,
        attr=attr,
        name=name,
        overlay=True,  # Ensures it can be toggled
        control=True,
        opacity=opacity  # Set transparency
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Display map
    return m