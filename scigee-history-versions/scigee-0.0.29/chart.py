import ee
import pandas as pd
from datetime import datetime

def gee2df(collection_name, lat, lon, date_range, bandname, scale, radius = 0.001):
    # unit degree, format:
    # [minlon, minlat,
    #  maxlon, maxlat]
    roi = ee.Geometry.Rectangle([
        lon - radius, lat - radius, 
        lon + radius, lat+ radius
    ])
    start_date, end_date = date_range

    collection = ee.ImageCollection(collection_name)\
        .filterBounds(roi).filterDate(start_date, end_date)
    # Sort the filtered collection by date in ascending order
    collection = collection.sort('system:time_start')

    def interp_image(image, bandname, scale):
        image = ee.Image(image)
        date = image.get('system:time_start')

        image_band = image.select(bandname)
        stats = image_band.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=scale)
        val = stats.get(bandname)
        return image.set('Info', [date, val])

    # Map the function to the image collection
    # collection = collection.map(interp_image)
    collection = collection.map(lambda image: interp_image(image, bandname, scale))

    # Use aggregate_array to get the values as an array
    array = collection.aggregate_array('Info')


    # Convert the array to a list using getInfo()
    array = array.getInfo()
    df = pd.DataFrame(array, columns = ['DATETIME', 'VALUE'])
    df['DATETIME'] = df['DATETIME'].map(
        lambda x: datetime.utcfromtimestamp(int(x) // 1000)
    )
    df = df.set_index('DATETIME')
    return df