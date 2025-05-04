import ee

class VI(object):
    def __init__(self):
        pass

    def __call__(self, image, vi, **kwargs):
        if vi == "ndvi":
            image = self.calc_ndvi(image, **kwargs)
        elif vi == "evi":
            image = self.calc_evi(image, **kwargs)
        elif vi == "cire":
            image = self.calc_cire(image, **kwargs)
        elif vi == "ndi":
            image = self.calc_ndi(image, **kwargs)

        return image

    @staticmethod
    def calc_ndvi(image, **kwargs):
        nir = kwargs["nir"]
        red = kwargs["red"]

        evi = image.expression(
            '((NIR - RED) / (NIR + RED))', {
            'NIR': image.select(nir),
            'RED': image.select(red)
        })
        return image.addBands(evi.rename("NDVI"))

    @staticmethod
    def calc_evi(image, **kwargs):
        nir = kwargs["nir"]
        red = kwargs["red"]
        blue = kwargs["blue"]

        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': image.select(nir),
            'RED': image.select(red),
            'BLUE': image.select(blue)
        })
        return image.addBands(evi.rename("EVI"))

    @staticmethod
    def calc_cire(image, **kwargs):
        re2 = kwargs["re2"]
        re3 = kwargs["re3"]

        cire = image.expression(
            '(RE3/RE2) - 1', {
            'RE2': image.select(re2),
            'RE3': image.select(re3)
        })
        return image.addBands(cire.rename("CIRE"))

    @staticmethod
    def calc_ndi(image, **kwargs):
        band_a = kwargs["b1"]
        band_b = kwargs["b2"]

        ndi = image.expression(
            '(band_a - band_b) / (band_a + band_b)', {
            'band_a': image.select(band_a),
            'band_b': image.select(band_b)
        })
        if "label" in kwargs.keys():
            label = kwargs["label"]
        else:
            label = "NDI"
        return image.addBands(ndi.rename(label))

class Point(object):
    def __init__(self):
        pass

    def __call__(self, bounds = None, lon = None, lat = None):
        if bounds:
            lon, lat = bounds
        assert lon, "No longitude input..."
        assert lat, "No latitude input..."
        point = ee.Geometry.Point(lon, lat)
        return point

    @classmethod
    def get_circle_buffer(self, point, buffer_size = 100):
        return point.buffer(buffer_size)

    @classmethod
    def get_rect_buffer(self, point, buffer_size = 0.5):
        # buffer_size unit eq proj unit, e.g., degree for WGS84
        lon, lat = point.getInfo()["coordinates"]
        # example: [-97.94, 26.81, -96.52, 26.84]
        bounds = [lon - buffer_size, lat - buffer_size, lon + buffer_size, lat + buffer_size]
        return ee.Geometry.Rectangle(bounds)

class Polygon(object):
    def __init__(self):
        pass

    def __call__(self, bounds):
        # for rectangele: bounds = [-97.94, 26.81, -96.52, 26.84] ## sample land / sea bounds
        # for polygon: bounds = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]

        if isinstance(bounds, list):
            if np.array(bounds).ndim == 1:
                region = ee.Geometry.Rectangle(bounds)
            else:
                region = ee.Geometry.Polygon(bounds)
        return region

class Geometry(Point, Polygon):
    def __init__(self, bounds = None, lon = None, lat = None):
        Point.__init__(self)
        Polygon.__init__(self)
        self.bounds = bounds
        self.lon = lon
        self.lat = lat

    def __call__(self, geom_type):
        if geom_type == 0 or geom_type == "point":
            geom = Point.__call__(self, bounds = self.bounds, lon = self.lon, lat = self.lat)
        elif geom_type == 2 or geom_type == "polygon":
            geom = Polygon.__call__(self, bounds = self.bounds)
        return geom


class Utils(VI):
    def __init__(self):
        VI.__init__(self)

    @classmethod
    def calc_vi(self, image, vi, **kwargs):
        return VI.__call__(self, image, vi, **kwargs)

    @classmethod
    def sentinel_2_cloud_mask(self, image, mask_out = True):
        """
        javascript code:

        # function maskS2clouds(image) {
        #   var qa = image.select('QA60');

        #   // Bits 10 and 11 are clouds and cirrus, respectively.
        #   var cloudBitMask = 1 << 10;
        #   var cirrusBitMask = 1 << 11;

        #   // Both flags should be set to zero, indicating clear conditions.
        #   var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
        #       .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

        #   return image.updateMask(mask).divide(10000);
        # }

        European Space Agency (ESA) clouds from 'QA60', i.e. Quality Assessment band at 60m
        
        parsed by Nick Clinton
        """

        qa = image.select('QA60')

        # bits 10 and 11 are clouds and cirrus
        cloudBitMask = int(2**10)
        cirrusBitMask = int(2**11)

        # both flags set to zero indicates clear conditions.
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(\
            qa.bitwiseAnd(cirrusBitMask).eq(0))
        
        if mask_out:
            return image.updateMask(mask)#.divide(10000)
        else:
            # clouds is not clear
            cloud = mask.Not().rename(['ESA_clouds'])

            # return the masked and scaled data.
            return image.addBands(cloud)

    # @classmethod
    # def coors2roi(self, bounds):
    #     # for rectangele: bounds = [-97.94, 26.81, -96.52, 26.84] ## sample land / sea bounds
    #     # for polygon: bounds = [[[105.532,19.059],[105.606,19.058],[105.605,19.108],[105.530,19.110],[105.532,19.059]]]
    #     if isinstance(bounds, list):
    #         if np.array(bounds).ndim == 1:
    #             roi = ee.Geometry.Rectangle(bounds)
    #         else:
    #             roi = ee.Geometry.Polygon(bounds)
    #     return roi

    # @classmethod
    # def coor2point(self, coors = None, lon = None, lat = None):
    #     if coors:
    #         lon, lat = coors
    #     assert lon, "No longitude input..."
    #     assert lat, "No latitude input..."
    #     point = ee.Geometry.Point(lon, lat)
    #     return point

    # @classmethod
    # def get_circle_buffer(self, point, buffer_size = 100):
    #     return point.buffer(buffer_size)

    # @classmethod	
    # def get_rect_buffer(self, point, buffer_size = 0.5):
    #     # buffer_size unit eq proj unit, e.g., degree for WGS84
    #     lon, lat = point.getInfo()["coordinates"]
    #     # example: [-97.94, 26.81, -96.52, 26.84]
    #     bounds = [lon - buffer_size, lat - buffer_size, lon + buffer_size, lat + buffer_size]
    #     return ee.Geometry.Rectangle(bounds)