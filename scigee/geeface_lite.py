import ee
import numpy as np
import pandas as pd
from . import chart

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

def timeseries2df(collection: ee.ImageCollection, bands: list, label_bands: list, lon: float, lat: float, date_range: list, scale: float) -> pd.DataFrame:
    """
    Search ee image colection covering point [lon: float, lat:float], and extract time series at the point by date_range ([start: str, end: str])
    Input:
    -------
    collection: ee image collection list
    bands: list, names (str) of bands for ee image
    label_banes: list, user specified names (str) for bands
    lon: float, longitude
    lat: flat, latitude
    date_range: list, [start: str, end: str]
    scale: numeric (float/int), pixel spatial resolution (m)
    Return:
    -------
    df: pandas dataframe time series
    """
    point = ee.Geometry.Point([lon, lat])

    col = ee.ImageCollection(collection).filterBounds(point)
    time_series = col.filterDate(*date_range)
 
    chart_ts = chart.Image.series(**{
        'imageCollection': time_series, 
        'region': point,
        'scale': scale,
        'bands': bands,
        'label_bands':label_bands,
        # 'properties':['CLOUD_COVERAGE_ASSESSMENT'],
        # 'label_properties':['CLOUD_COVER']
    })
 
    df = chart_ts.dataframe
    return df

def region2array(image: ee.Image, roi: ee.Geometry, scale: float, max_pixels: int = 1e9, mask_value: float = -9999) -> pd.DataFrame:
    """
    Get pixel values and coordinates of a region as a dataframe.
    This method converts image with coordinates to dataframe directly, user memory issue may be more likely to occur.
    """
    image = image.addBands(ee.Image.pixelLonLat())
    image = image.unmask(mask_value)
    # ee.Image -> ee.dictionary.Dictionary
    image = image.reduceRegion(
        reducer = ee.Reducer.toList(),
        # crs = 'EPSG:4326',
        geometry = roi,
        maxPixels = 1e13,
        scale = scale
    )
    df = pd.DataFrame.from_dict(image.getInfo(), orient = 'columns')
    return df

def region2arrayB(image: ee.Image, roi: ee.Geometry, band: str, scale: float, max_pixels: int = 1e9, mask_value: float = -9999) -> pd.DataFrame:
    """
    Get pixel values and coordinates of a region as a dataframe.
    Extract pixles values, longitudes, and latitudes seprately, as a plan B for region2array.
    """
    image = image.addBands(ee.Image.pixelLonLat())
    image = image.unmask(mask_value)
    # ee.Image -> ee.dictionary.Dictionary
    image = image.reduceRegion(
        reducer = ee.Reducer.toList(),
        # crs = 'EPSG:4326',
        geometry = roi,
        maxPixels = 1e13,
        scale = scale
    )
    data = np.array((ee.Array(image.get(band)).getInfo()))
    lats = np.array((ee.Array(image.get("latitude")).getInfo()))
    lons = np.array((ee.Array(image.get("longitude")).getInfo()))
    df = pd.DataFrame(zip(data, lats, lons), columns = [band, "latitude", "longitude"])
    return df

def region2arrayC(image: ee.Image, roi: ee.Geometry, bands: list, scale: float, max_pixels: int = 1e9, mask_value: float = -9999) -> pd.DataFrame:
    """
    Get pixel values and coordinates of a region as a dataframe.
    Extract pixles values, longitudes, and latitudes seprately, as a plan B for region2array.
    """
    image = image.addBands(ee.Image.pixelLonLat())
    image = image.unmask(mask_value)
    # ee.Image -> ee.dictionary.Dictionary
    image = image.reduceRegion(
        reducer = ee.Reducer.toList(),
        # crs = 'EPSG:4326',
        geometry = roi,
        maxPixels = 1e13,
        scale = scale
    )
    data_list = []
    for band in bands:
        data = np.array((ee.Array(image.get(band)).getInfo()))
        data_list.append(data)
    lats = np.array((ee.Array(image.get("latitude")).getInfo()))
    lons = np.array((ee.Array(image.get("longitude")).getInfo()))
    df = pd.DataFrame(zip(*data_list, lats, lons), columns = bands + ["latitude", "longitude"])
    return df

def region2array_obs(image: ee.Image, band: str or list, roi: ee.Geometry, proj: ee.Projection, scale: float, max_pixels: int = 1e9, pick_coords: bool = False):
    """
    Obsolete region2array function, will be deprecated.
    """
    if isinstance(band, str):
        band = [band]
    image_array = image.reproject(proj).select(band).reduceRegion(reducer = ee.Reducer.toList(), geometry = roi, scale = scale, maxPixels = max_pixels).getInfo()
    df = pd.DataFrame(list(image_array.values())).T
    df.columns = list(image_array.keys())
    coords = image.pixelLonLat().reproject(proj)
    coords = coords.select(['longitude', 'latitude']).reduceRegion(reducer = ee.Reducer.toList(), geometry = roi, scale = scale, maxPixels = max_pixels).getInfo()
    if pick_coords:
        len_coords = len(coords["longitude"])
        assert len_coords == len(coords["latitude"]), "size of longitude and latitude are different"
        if len_coords > len(df):
            selected_index = np.sort(np.random.choice(np.arange(len_coords), len(df), replace=False))
            coords["longitude"] = np.array(coords["longitude"])[selected_index]
            coords["latitude"] = np.array(coords["latitude"])[selected_index]
        else:
            selected_index = np.sort(np.random.choice(np.arange(len(df)), len_coords, replace=False))
            df = df.iloc[selected_index, :]
    df["lons"] = coords["longitude"]
    df["lats"] = coords["latitude"]
    return df

def draw_image_obs(image: ee.Image, band: str or list, roi: ee.Geometry, to8bit: bool = False, color_order: list = None) -> None:
    """
    Obsolete draw_image function, will be deprecated
    Example:
    import ee
    import numpy as np
    import matplotlib.pyplot as plt
 
 
    # Define an image.
    img = ee.Image('LANDSAT/LC08/C01/T1_SR/LC08_038029_20180810') \
    .select(['B4', 'B5', 'B6'])
 
    # Define an area of interest.
    aoi = ee.Geometry.Polygon(
    [[[-110.8, 44.7],
        [-110.8, 44.6],
        [-110.6, 44.6],
        [-110.6, 44.7]]], None, False)
 
    draw_image(img, ["B4", "B5", "B6"], aoi, to8bit = True, color_order = [2, 1, 0])
    """
    samples = image.sampleRectangle(region=roi)
    if isinstance(band, list):
        array = [np.array(samples.get(b).getInfo())[:, :, np.newaxis] for b in band]
        array = np.concatenate(array, axis = 2) # bands order
    else:
        array = np.array(samples.get(band).getInfo())
    # Scale the data to [0, 255] to show as an RGB image.
    if to8bit:
        array = (255*((array - 100)/3500)).astype('uint8')
        if color_order: 
            array = array[:, :, color_order]
    plt.imshow(array)
    plt.show()

def image2drive(collection_name, band_names, roi_bound, date_range, savefolder, scale, crs = 'EPSG:4326'):
    minlon, minlat, maxlon, maxlat = roi_bound
    start_date, end_date = date_range
    roi = ee.Geometry.BBox(minlon, minlat, maxlon, maxlat)

    if type(collection_name) == ee.ImageCollection:
        collection = collection_name
    elif type(collection_name) == str:
        collection = (
            ee.ImageCollection(collection_name)
            .filterBounds(roi)
            .filterDate(start_date, end_date)
        )
    else:
        raise Exception("ERROR: Wrong collection type, it must of 'str' or 'ee.ImageCollection'") 
    image = collection.select(band_names).reduce(ee.Reducer.mean())

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=start_date,
        folder=savefolder,
        region=roi,
        scale=scale,
        maxPixels = 1e13,
        crs=crs
    )
    task.start()

def get_status():
    tasks = ee.data.listOperations()
    pending_tasks = sum(1 for task in tasks if task['metadata']['state'] == 'PENDING')
    return pending_tasks

class DataInfo:
    def __init__(self):
      pass
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def s5p_o3(NTRI = True): # start from 2018-09-08
        if NTRI:
            collection = "COPERNICUS/S5P/NRTI/L3_O3"
        else:
            collection = "COPERNICUS/S5P/OFFL/L3_O3"
        bands = [
                "O3_column_number_density", "O3_column_number_density_amf", "O3_slant_column_number_density",
                "O3_effective_temperature", "cloud_fraction", 
                "sensor_azimuth_angle", "sensor_zenith_angle", "solar_azimuth_angle", "solar_zenith_angle"
        ]
        label_bands = [
                    "o3_vcd", "o3_amf", "o3_scd",
                    "o3_temperature", "s5p_cloud_fraction",
                    "s5p_sen_azi", "s5p_sen_zen", "s5p_sun_azi", "s5p_sun_zen"
        ]
        scale = 0.01 * 100 * 1000 # 0.01 arc degree -> meters
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def s5p_hcho(NTRI = True): # start from 2018-10-02
        if NTRI:
            collection = "COPERNICUS/S5P/NRTI/L3_HCHO"
        else:
            collection = "COPERNICUS/S5P/OFFL/L3_HCHO"
        bands = [
                "tropospheric_HCHO_column_number_density", "tropospheric_HCHO_column_number_density_amf", "HCHO_slant_column_number_density",
                "cloud_fraction",
                "sensor_azimuth_angle", "sensor_zenith_angle", "solar_azimuth_angle", "solar_zenith_angle"
        ]
        label_bands = [
                    "hcho_vcd", "hcho_amf", "hcho_scd",
                    "s5p_cloud_fraction",
                    "s5p_sen_azi", "s5p_sen_zen", "s5p_sun_azi", "s5p_sun_zen"
        ]
        scale = 0.01 * 100 * 1000 # 0.01 arc degree -> meters
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def s5p_no2(NTRI = True): # start from 2018-07-10
        if NTRI:
            collection = "COPERNICUS/S5P/NRTI/L3_NO2"
        else:
            collection = "COPERNICUS/S5P/OFFL/L3_NO2"
        bands = [
                "NO2_column_number_density", "tropospheric_NO2_column_number_density", "stratospheric_NO2_column_number_density",
                "NO2_slant_column_number_density", "tropopause_pressure", "absorbing_aerosol_index", "cloud_fraction",
                "sensor_altitude", "sensor_azimuth_angle", "sensor_zenith_angle", "solar_azimuth_angle", "solar_zenith_angle"
        ]
        label_bands = [
                    "no2_total_vcd", "no2_vcd", "no2_strato_vcd", "no2_scd",
                    "tropopause_pressure", "absorbing_aerosol_index", "s5p_cloud_fraction",
                    "sp5_sen_alt", "s5p_sen_azi", "s5p_sen_zen", "s5p_sun_azi", "s5p_sun_zen"
        ]
        scale = 0.01 * 100 * 1000 # 0.01 arc degree -> meters
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def s5p_co(NTRI = True): # start from 2018-06-28
        if NTRI:
            collection = "COPERNICUS/S5P/NRTI/L3_CO"
        else:
            collection = "COPERNICUS/S5P/OFFL/L3_CO"
        bands = [
                "CO_column_number_density", "H2O_column_number_density", "cloud_height",
                "sensor_altitude", "sensor_azimuth_angle", "sensor_zenith_angle", "solar_azimuth_angle", "solar_zenith_angle"
        ]
        label_bands = [
                    "co_vcd", "h2o_vcd", "cloud_height",
                    "sp5_sen_alt", "s5p_sen_azi", "s5p_sen_zen", "s5p_sun_azi", "s5p_sun_zen"
        ]
        scale = 0.01 * 100 * 1000 # 0.01 arc degree -> meters
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def s5p_so2(NTRI = True): # start from 2018-07-10
        if NTRI:
            collection = "COPERNICUS/S5P/NRTI/L3_SO2"
        else:
            collection = "COPERNICUS/S5P/OFFL/L3_SO2"
        bands = [
                "SO2_column_number_density", "SO2_column_number_density_amf", "SO2_slant_column_number_density",
                "absorbing_aerosol_index", "cloud_fraction",
                "sensor_azimuth_angle", "sensor_zenith_angle", "solar_azimuth_angle", "solar_zenith_angle",
                "SO2_column_number_density_15km"
        ]
        label_bands = [
                    "so2_vcd", "so2_amf", "hcho_scd",
                    "s5p_aai", "s5p_cloud_fraction",
                    "s5p_sen_azi", "s5p_sen_zen", "s5p_sun_azi", "s5p_sun_zen",
                    "so2_vcd_15km"
        ]
        scale = 0.01 * 100 * 1000 # 0.01 arc degree -> meters
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def s5p_ch4(NTRI = True): # start from 2019-02-08
        if NTRI:
            collection = "COPERNICUS/S5P/NRTI/L3_CH4"
        else:
            collection = "COPERNICUS/S5P/OFFL/L3_CH4"
        bands = [
            "CH4_column_volume_mixing_ratio_dry_air", 
            "aerosol_height", "aerosol_optical_depth",
            "sensor_azimuth_angle", "sensor_zenith_angle",
            "solar_azimuth_angle", "solar_zenith_angle"
        ]
        label_bands = [
                    "xch4", "aerosol_height", "aod", 
                    "s5p_sen_azi", "s5p_sen_zen", 
                    "s5p_sun_azi", "s5p_sun_zen"
        ]
        scale = 0.01 * 100 * 1000 # 0.01 arc degree -> meters
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def s5p_aai(NTRI = True): # start from 2018-07-04
        if NTRI:
            collection = "COPERNICUS/S5P/NRTI/L3_AER_AI"
        else:
            collection = "COPERNICUS/S5P/OFFL/L3_AER_AI"
        bands = [
            "absorbing_aerosol_index", "sensor_altitude", "sensor_azimuth_angle", "sensor_zenith_angle",
            "solar_azimuth_angle", "solar_zenith_angle"
        ]
        label_bands = [
                    "aai", "sp5_sen_alt", "s5p_sen_azi", "s5p_sen_zen", 
                    "s5p_sun_azi", "s5p_sun_zen"
        ]
        scale = 0.01 * 100 * 1000 # 0.01 arc degree -> meters
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def modis_reflectance():
        collection = "MODIS/006/MCD43A4" # 1-Day
        bands = [
            'Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band3',
            'Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band5', 'Nadir_Reflectance_Band6',
            'Nadir_Reflectance_Band7', 'BRDF_Albedo_Band_Mandatory_Quality_Band1'
        ]
        label_bands = [
            'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'qc'
        ]
        scale = 500
        return collection, bands, label_bands, scale
    # -------------------------------------------------------------------------------------------------------------- 
    @staticmethod
    def modis_aod():
        collection = "MODIS/006/MCD19A2_GRANULES" # 1-Day
        bands = [
                'Optical_Depth_055', 'Optical_Depth_047'
        ]
        label_bands = [
                    'AOD055', 'AOD047'
        ]
        scale = 1000
        return collection, bands, label_bands, scale
    # -------------------------------------------------------------------------------------------------------------- 
    @staticmethod
    def modis_lai_fpar(): # 4-Day
        collection = "MODIS/006/MCD15A3H"
        bands = ['Fpar', 'Lai', "FparLai_QC"]
        label_bands = ['Fpar', 'Lai', "FparLai_QC"]
        scale = 500
        return collection, bands, label_bands, scale
    # -------------------------------------------------------------------------------------------------------------- 
    @staticmethod
    def modis_gpp(sat = "Terra"):
        if sat == "Terra":
            collection = "MODIS/006/MOD17A2H" # 8-Day # Terra; Aqua: MODIS/006/MYD17A2H
        else:
            collection = "MODIS/006/MYD17A2H" # 8-Day Aqua
        bands = [
                'Gpp', 'PsnNet'
        ]
        label_bands = [
                'GPP', 'NEE'
        ]
        scale = 500
        return collection, bands, label_bands, scale
    # -------------------------------------------------------------------------------------------------------------- 
    @staticmethod
    def modis_evi(sat = "Terra"):
        if sat == "Terra":
            collection = "MODIS/MOD09GA_006_EVI" # Daily Terra
        else:
            collection = "MODIS/MYD09GA_006_EVI" # Daily Aqua
        bands = [
                'EVI',
        ]
        label_bands = [
                'EVI'
        ]
        scale = 500
        return collection, bands, label_bands, scale
    # -------------------------------------------------------------------------------------------------------------- 
    @staticmethod
    def modis_ndvi(sat = "Terra"):
        if sat == "Terra":
            collection = "MODIS/MOD09GA_006_NDVI" # Daily Terra
        else:
            collection = "MODIS/MYD09GA_006_NDVI" # Daily Aqua
        bands = [
                'NDVI',
        ]
        label_bands = [
                'NDVI'
        ]
        scale = 500
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def modis_net_evapo(): # net_evapotranspiration
        collection = "MODIS/006/MOD16A2" # Terra Net Evapotranspiration 8-Day Global 500m
        bands = [
                'ET', 'LE', 'PET', 'PLE', 'ET_QC'
        ]
        label_bands = [
            'ET', 'LE', 'PET', 'PLE', 'ET_QC'
        ]
        scale = 500
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def landsat7_evi(nday = 8):
        if nday == 8:
            collection = "LANDSAT/LE07/C01/T1_8DAY_EVI" # 8-Day
        elif nday == 32:
            collection = "LANDSAT/LE07/C01/T1_32DAY_EVI" # 32-Day
        else:
            collection = "LANDSAT/LE07/C01/T1_ANNUAL_EVI" # Annual
        bands = [
            'EVI'
        ]
        label_bands = [
            'EVI'
        ]
        scale = 30
        return collection, bands, label_bands, scale

    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def landsat7_ndvi(nday = 8):
        if nday == 8:
            collection = "LANDSAT/LE07/C01/T1_8DAY_NDVI" # 8-Day
        elif nday == 32:
            collection = "LANDSAT/LE07/C01/T1_32DAY_NDVI" # 32-Day
        else:
            collection = "LANDSAT/LE07/C01/T1_ANNUAL_NDVI" # Annual
        bands = [
            'NDVI'
        ]
        label_bands = [
            'NDVI'
        ]
        scale = 30
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def landsat8_evi(nday = 8):
        if nday == 8:
            collection = "LANDSAT/LC08/C01/T1_8DAY_EVI" # 8-Day
        elif nday == 32:
            collection = "LANDSAT/LC08/C01/T1_32DAY_EVI" # 32-Day
        else:
            collection = "LANDSAT/LC08/C01/T1_ANNUAL_EVI" # Annual
        bands = [
            'EVI'
        ]
        label_bands = [
            'EVI'
        ]
        scale = 30
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def landsat8_ndvi(nday = 8):
        if nday == 8:
            collection = "LANDSAT/LC08/C01/T1_8DAY_NDVI" # 8-Day
        elif nday == 32:
            collection = "LANDSAT/LC08/C01/T1_32DAY_NDVI" # 32-Day
        else:
            collection = "LANDSAT/LC08/C01/T1_ANNUAL_NDVI" # Annual
        bands = [
            'NDVI'
        ]
        label_bands = [
            'NDVI'
        ]
        scale = 30
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def landsat7_srt1():
        collection = "LANDSAT/LE07/C02/T1_L2"
        bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "sr_atmos_opacity", "sr_cloud_qa"]
        label_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "sr_atmos_opacity", "sr_cloud_qa"]
        scale = 30
        return collection, bands, label_bands, scale    
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def landsat8_srt1():
        collection = "LANDSAT/LC08/C02/T1_L2"
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11', 'sr_aerosol']
        label_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11', 'sr_aerosol']
        scale = 30
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_cfsv2():
        collection = 'NOAA/CFSV2/FOR6H' # 6-Hourly
        bands = [
                'Temperature_height_above_ground', 'Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average', 'Volumetric_Soil_Moisture_Content_depth_below_surface_layer_5_cm',
                'Maximum_specific_humidity_at_2m_height_above_ground_6_Hour_Interval', 'Minimum_specific_humidity_at_2m_height_above_ground_6_Hour_Interval',
                'u-component_of_wind_height_above_ground', 'v-component_of_wind_height_above_ground', 'Precipitation_rate_surface_6_Hour_Average'
        ]
        label_bands = [
                "TAIR", "RG", "SMP", 
                "SHMAX", "SHMIN", "UWIND", "VWIND", "PRECIP"
        ]
        scale = 0.2*100*1000
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_cloud():
        collection = "NOAA/CDR/PATMOSX/V53"
        bands = ["cloud_fraction", "cloud_fraction_uncertainty", "cloud_probability"]
        label_bands = ["cloud_fraction", "cloud_fraction_uncertainty", "cloud_probability"]
        scale = 0.1 * 100 * 1000
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_toms_omi_ozone():
        collection = "TOMS/MERGED"
        bands = ["ozone"]
        label_bands = ["ozone"]
        scale = 1 * 100 * 1000
        return collection, bands, label_bands, scale
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def pml_v2_water():
        collection = "CAS/IGSNRR/PML/V2"
        bands = ["GPP", "Ec", "Es", "Ei", "ET_water"]
        label_bands = ["GPP", "transpiration", "evaporation", "Interception_from_vegetation_canopy", "ET_water"]
        scale = 500
        return collection, bands, label_bands, scale

    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def VIIRS_nightlight_corrected():
        collection = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG" # 1-Day
        bands = [
            "avg_rad", # nanoWatts/cm2/sr; Average DNB radiance values.
            "cf_cvg" # Cloud-free coverages; the total number of observations that went into each pixel. 
                    # This band can be used to identify areas with low numbers of observations where the quality is reduced.
        ]
        label_bands = [
            "avg_rad",
            "cf_cvg"
        ]
        scale = 450 # 15 arc seconds 1 arc second = 30 meters
        return collection, bands, label_bands, scale

    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    # Sentinel-2 MSI Level-1C
    def s2_msi1C():
        collection = "COPERNICUS/S2"
        bands = [
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
            'B8', 'B8A', 'B9', 'B10', 'B11', 'B12',
            'QA10', 'QA20', 'QA60'
        ]
        label_bands = [
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
            'B8', 'B8A', 'B9', 'B10', 'B11', 'B12',
            'QA10', 'QA20', 'QA60'
        ]

        scale = 10 # 10 m for most bands
        return collection, bands, label_bands, scale

    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    # Sentinel-2 MSI Level-2A
    def s2_msi2A():
        collection = "COPERNICUS/S2_SR"
        bands = [
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
            'B8', 'B8A', 'B9', 'B11', 'B12',
            'AOT', 'WVP', 'SCL', 'TCI_R', 'TCI_G', 'TCI_B',
            'MSK_CLDPRB', 'MSK_SNWPRB', 
            'QA10', 'QA20', 'QA60'
        ]
        label_bands = [
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
            'B8', 'B8A', 'B9', 'B11', 'B12',
            'AOT', 'WVP', 'SCL', 'TCI_R', 'TCI_G', 'TCI_B',
            'MSK_CLDPRB', 'MSK_SNWPRB', 
            'QA10', 'QA20', 'QA60'
        ]

        scale = 10 # 10 m for most bands
        return collection, bands, label_bands, scale
    
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    # Sentinel-3 OLCI dataset
    def s3_olci():
        collection = "COPERNICUS/S3/OLCI"
        bands = [
                'Oa01_radiance', 'Oa02_radiance', 'Oa03_radiance', 'Oa04_radiance', 'Oa05_radiance',
                'Oa06_radiance', 'Oa07_radiance', 'Oa08_radiance', 'Oa09_radiance', 'Oa10_radiance',
                'Oa11_radiance', 'Oa12_radiance', 'Oa13_radiance', 'Oa14_radiance', 'Oa15_radiance',
                'Oa16_radiance', 'Oa17_radiance', 'Oa18_radiance', 'Oa19_radiance', 'Oa20_radiance', 'Oa21_radiance'
        ]
        label_bands = ['oa01', 'oa02', 'oa03', 'oa04', 'oa05',
                    'oa06', 'oa07', 'oa08', 'oa09', 'oa10',
                    'oa11', 'oa12', 'oa13', 'oa14', 'oa15',
                    'oa16', 'oa17', 'oa18', 'oa19', 'oa20', 'oa21'
        ]

        # rad_scale * band -> radiace
        # in order: band1 -> band21
        rad_scale = [
                    0.0139465, 0.0133873, 0.0121481, 0.0115198, 0.0100953,
                    0.0123538, 0.00879161, 0.00876539, 0.0095103, 0.00773378,
                    0.00675523, 0.0071996, 0.00749684, 0.0086512, 0.00526779,
                    0.00530267, 0.00493004, 0.00549962, 0.00502847, 0.00326378, 0.00324118
        ]
        scale = 300
        return collection, bands, label_bands, scale, rad_scale

    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def AVHRR_reflectance():
        collection = "NOAA/CDR/AVHRR/SR/V5"
        bands = [
            'SREFL_CH1', 'SREFL_CH2', 'SREFL_CH3', 'BT_CH3', 'BT_CH4', 'BT_CH5'
        ]
        label_bands = [
            'SREFL_CH1', 'SREFL_CH2', 'SREFL_CH3', 'BT_CH3', 'BT_CH4', 'BT_CH5'
        ]
        scale = 5566
        return collection, bands, label_bands, scale

    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def ERA5_hourly():
        # ERA5-Land Hourly - ECMWF Climate Reanalysis
        collection = "ECMWF/ERA5_LAND/HOURLY"
        bands  = [
            'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards',
            'dewpoint_temperature_2m', 'temperature_2m', 'soil_temperature_level_1', 
            'u_component_of_wind_10m', 'v_component_of_wind_10m', 
            'total_precipitation', 'surface_pressure',
        ]
        label_bands = [
            'SWIN', 'LWIN', 'D2m', 'T2m', 'Tsoil1', 'U', 'V', 'P', 'PA'
        ]
        scale = 11132
        return collection, bands, label_bands, scale

    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def modis_lc(): # 2001-01-01T00:00:00Z - 2019-01-01T00:00:00
        collection = "MODIS/006/MCD12Q1"
        bands = [
            "LC_Type1", "LC_Type2", "LC_Type3", "LC_Type4", "LC_Type5", 
            "LC_Prop1_Assessment", "LC_Prop2_Assessment", "LC_Prop3_Assessment", 
            "LC_Prop1", "LC_Prop2", "LC_Prop3", 
            "QC", "LW"
        ]
        label_bands = [
            "LC_Type1", "LC_Type2", "LC_Type3", "LC_Type4", "LC_Type5", 
            "LC_Prop1_Assessment", "LC_Prop2_Assessment", "LC_Prop3_Assessment", 
            "LC_Prop1", "LC_Prop2", "LC_Prop3", 
            "QC", "LW"
        ]
        scale = 500 # meters
        return collection, bands, label_bands, scale