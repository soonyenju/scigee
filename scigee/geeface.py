import os
import ee
import requests
import numpy as np

class EEarth():
    """
    Earth engine object
    """
    def __init__(self, source):
        self.source = source

    def fetch_image(self):
        self.image = ee.Image(self.source)

    def fetch_collection(self, date_range = [], roi = None):
        collection = ee.ImageCollection(self.source)
        # filter by date:
        if date_range:
            collection = collection.filterDate(*date_range)
        # filter by bounds:
        if roi:
            collection = collection.filterBounds(roi)
        self.__length__ = collection.size().getInfo()
        self.collection = collection

class Ecolbox(EEarth):
    """
    Ecolbox: Earth engine Collection Toolbox
    """
    def __init__(self, collection_name, date_range = [], roi = None):
        EEarth.__init__(self, collection_name)
        self.fetch_collection(date_range, roi)
        self.__length__ = self.collection.size().getInfo()

    def __call__(self, collection):
        # update collection box
        self.__length__ = collection.size().getInfo()
        self.collection = collection

    # select certain image
    def __getitem__(self, idx):
        collection = self.collection
        if not isinstance(collection, ee.ee_list.List):
            self.__to_ee_list()
            image = ee.Image(self.__image_list__.get(idx))
        return image

    def __to_ee_list(self):
        self.__image_list__ = self.collection.toList(self.collection.size())

    def to_list(self, collection):
        if not collection:
            collection = self.collection
        return collection.toList(collection.size()).getInfo()

    def get_image_by_index(self, collection = None, idx = 0):
        # default: first.
        if collection:
            collection = collection.toList(collection.size())
            image = ee.Image(collection.get(idx))
        else:
            collection = self.collection
            if not isinstance(collection, ee.ee_list.List):
                self.__to_ee_list()
                image = ee.Image(self.__image_list__.get(idx))
        return image

    def fmap(self, func, *args, auto_update = True, **kwargs):
        # simplify the map function in the future
        if auto_update:
            self.__call__(
                self.collection.map(
                    lambda image: func(image, *args, **kwargs)
                )
            )
        else:
            return self.collection.map(
                lambda image: func(image, *args, **kwargs)
            )

    def reduce_collection(self, collection = None, band = None, label = "DATA", band_reducer = ee.Reducer.mean(), spatial_reducer = ee.Reducer.median()):
        if not collection:
            collection = self.collection
        if band:
            image = collection.select(band).reduce(spatial_reducer).rename(band)
        else:
            image = collection.map(
                    lambda image: image.reduce(band_reducer)
                ).reduce(spatial_reducer).rename(label)
        return image

class Emagebox(object):
    """
    Emagebox: Earth engine Image Toolbox
    """
    def __init__(self, image, scale = 30, max_pixels = 1e8, default_value = -9999):
        self.image = image
        self.scale = scale
        self.max_pixels = max_pixels
        self.default_value = default_value
    
    # select certain band
    def __getitem__(self, band_name):
        return self.image.select(band_name)

    def set_scale(self, scale):
        self.scale = scale
    
    def set_max_pixels(self, max_pixels):
        self.max_pixels = max_pixels
    
    def set_default_value(self, default_value):
        self.default_value = default_value

    def get_band_names(self):
        return self.image.bandNames().getInfo()

    def get_stats(self, roi, reducer = ee.Reducer.mean()):
        stat = self.image.reduceRegion(
            reducer = reducer,
            geometry = roi,
            scale = self.scale,
            maxPixels = self.max_pixels
        )
        return stat.getInfo()
    
    def get_date(self):
        date = ee.Date(self.image.get('system:time_start'))
        return date.format('Y-M-d').getInfo()
    
    def get_proj(self, band, get_info = False):
        # // Get projection information from band.
        proj = self.image.select(band).projection()
        if get_info:
            proj = proj.getInfo()
        return proj

    def get_scale(self, band):
        # // Get scale (in meters) information from band.
        scale = self.image.select(band).projection().nominalScale()
        return scale.getInfo()

    def reproject(self, proj):
        return self.image.reproject(proj)

    def clip(self, roi):
        return self.image.clip(roi)

    def mask_value(self):
        mask = self.image.eq(self.default_value)
        self.image = self.image.updateMask(mask)

    def unmask(self):
        self.image = self.image.unmask(ee.Image.constant(self.default_value))

    def get_value(self, band, point, reducer = ee.Reducer.first()):
        value = self.image.select(band).reduceRegion(reducer, point, self.scale).get(band)
        value = ee.Number(value)
        return value.getInfo()

    def get_values(self, band, roi):
        # getInfo() is limited to 5000 records
        # ee.ee_exception.EEException: Array: No numbers in 'values', must provide a type.
        # see more at: https://gis.stackexchange.com/questions/321560/getting-dem-values-as-numpy-array-in-earth-engine
        # for rectangele: bounds = [-97.94, 26.81, -96.52, 26.84] ## sample land / sea bounds
        # for polygon: bounds = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]
        latlng = ee.Image.pixelLonLat().addBands(self.image.clip(roi).unmask(ee.Image.constant(self.default_value)))
        latlng = latlng.reduceRegion(
            reducer = ee.Reducer.toList(), 
            geometry = roi, 
            maxPixels = self.max_pixels, 
            scale = self.scale
        )
        lats = np.array((ee.Array(latlng.get("latitude")).getInfo()))
        lons = np.array((ee.Array(latlng.get("longitude")).getInfo()))
        try:
            values = np.array((ee.Array(latlng.get(band)).getInfo()))
        except:
            values = np.full_like(lats, np.nan, dtype = float)
        # self.values = list(values) ## print as list to check

        if not (values.shape == lats.shape == lons.shape):
            raise Exception(
                f"SizeError: " +
                f"values shape is {values.shape}, " + 
                f"lats shape is {lats.shape}, " + 
                f"lons shape is {lons.shape}."
            )
        return {
            "values":  values,
            "lons": lons,
            "lats": lats
        }

    def localize(self, save_name, save_folder = ".", crs_epsg = "4326"):
        url = self.image.getDownloadURL({
            "name": save_name,
            "crs": "EPSG:" + crs_epsg,
            "scale": self.scale
        })
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        save_dir = f"{save_folder}/{save_name}.zip"

        # Download the subset
        r = requests.get(url, stream = True)
        with open(save_dir, 'wb') as fd:
            for chunk in r.iter_content(chunk_size = 1024):
                fd.write(chunk)