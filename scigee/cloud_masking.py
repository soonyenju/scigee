import ee

def mask_clouds(image, method="cloud_prob", prob=60, maskCirrus=True, maskShadows=True, 
                scaledImage=False, dark=0.15, cloudDist=1000, buffer=250, cdi=None):
    # See: https://ee-extra.readthedocs.io/en/latest/_modules/ee_extra/QA/clouds.html
    """
    Masks clouds and shadows in an image or image collection for various satellite products.
    """
    
    def mask_VNP09GA(img):
        qf1 = img.select("QF1")
        qf2 = img.select("QF2")
        notCloud = qf1.bitwiseAnd(1 << 2).eq(0)
        if maskShadows:
            notCloud = notCloud.And(qf2.bitwiseAnd(1 << 3).eq(0))
        if maskCirrus:
            notCloud = notCloud.And(qf2.bitwiseAnd(1 << 6).eq(0))
            notCloud = notCloud.And(qf2.bitwiseAnd(1 << 7).eq(0))
        return img.updateMask(notCloud)

    def mask_S2(img):
        if method == "cloud_prob":
            clouds = ee.Image(img.get("cloud_mask")).select("probability")
            isCloud = clouds.gte(prob).rename("CLOUD_MASK")
        else:  # QA method
            qa = img.select("QA60")
            cloudBitMask = 1 << 10
            isCloud = qa.bitwiseAnd(cloudBitMask).eq(0)
            if maskCirrus:
                cirrusBitMask = 1 << 11
                isCloud = isCloud.And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        isCloud = isCloud.Not().rename("CLOUD_MASK")
        img = img.addBands(isCloud)

        if maskShadows:
            notWater = img.select("SCL").neq(6)
            darkPixels = img.select("B8").lt(dark * 1e4).multiply(notWater)
            shadowAzimuth = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
            cloudProjection = img.select("CLOUD_MASK").directionalDistanceTransform(shadowAzimuth, cloudDist / 10)
            cloudProjection = cloudProjection.reproject(crs=img.select(0).projection(), scale=10).select("distance").mask()
            isShadow = cloudProjection.multiply(darkPixels).rename("SHADOW_MASK")
            img = img.addBands(isShadow)

        cloudShadowMask = img.select("CLOUD_MASK")
        if maskShadows:
            cloudShadowMask = cloudShadowMask.add(img.select("SHADOW_MASK")).gt(0)
        cloudShadowMask = cloudShadowMask.focal_min(20, units="meters").focal_max(buffer * 2 / 10, units="meters").rename("CLOUD_SHADOW_MASK")
        img = img.addBands(cloudShadowMask)

        return img.updateMask(img.select("CLOUD_SHADOW_MASK").Not())

    def mask_L8(img):
        qa = img.select("pixel_qa")
        notCloud = qa.bitwiseAnd(1 << 5).eq(0)
        if maskShadows:
            notCloud = notCloud.And(qa.bitwiseAnd(1 << 3).eq(0))
        if maskCirrus:
            notCloud = notCloud.And(qa.bitwiseAnd(1 << 2).eq(0))
        return img.updateMask(notCloud)

    def mask_MOD09GA(img):
        qa = img.select("state_1km")
        notCloud = qa.bitwiseAnd(1 << 0).eq(0)
        if maskShadows:
            notCloud = notCloud.And(qa.bitwiseAnd(1 << 2).eq(0))
        if maskCirrus:
            notCloud = notCloud.And(qa.bitwiseAnd(1 << 8).eq(0))
        return img.updateMask(notCloud)

    # Mapping platforms to corresponding mask functions
    lookup = {
        "COPERNICUS/S2_SR": mask_S2,
        "LANDSAT/LC08/C01/T1_SR": mask_L8,
        "MODIS/006/MOD09GA": mask_MOD09GA,
        "NOAA/VIIRS/001/VNP09GA": mask_VNP09GA,
    }
