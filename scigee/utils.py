# def landsat_cloud_mask_toa(image):
#     # Add a cloud score band.  It is automatically called 'cloud'.
#     scored = ee.Algorithms.Landsat.simpleCloudScore(image)

#     # Create a mask from the cloud score and combine it with the image mask.
#     mask = scored.select(['cloud']).lte(20)

#     # Apply the mask to the image and display the result.
#     masked = image.updateMask(mask)
#     return masked

# Applies scaling factors.
def landsat_apply_scale_factors(image):
  optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
  thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
  return image.addBands(optical_bands, None, True).addBands(
      thermal_bands, None, True
  )

def mask_landsatsr_clouds(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = 1 << 3
    cloudsBitMask = 1 << 5
    # Get the pixel QA band.
    qa = image.select('QA_PIXEL')
    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloudShadowBitMask)
        .eq(0)
        .And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    )
    return image.updateMask(mask)


# def mask_landsatsr_clouds2(image):
#     '''
#     Seem not good as mask_landsatsr_clouds
#     This one has negative NDVI
#     '''
#     # select pixel_qa band as mask
#     qa = image.select('QA_PIXEL');

#     # conditions which to mask out - no shadows, snow or clouds
#     mask = (
#         qa.neq(68)
#         .And(qa.neq(132)).And(qa.neq(72)).And(qa.neq(136))
#         .And(qa.neq(80)).And(qa.neq(112)).And(qa.neq(144)).And(qa.neq(176))
#         .And(qa.neq(96)).And(qa.neq(160)).And(qa.neq(176)).And(qa.neq(224))
#     )


#     # apply mask
#     return image.updateMask(mask)

def mask_sentinel2sr_clouds(image):
  """Masks clouds in a Sentinel-2 image using the QA band.

  Args:
      image (ee.Image): A Sentinel-2 image.

  Returns:
      ee.Image: A cloud-masked Sentinel-2 image.
  """
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )

  return image.updateMask(mask).divide(10000)


# ==============================================================================
footprint_size = {
    'BSV': 100,
    'CRO': 200,
    'CSH': 200,
    'CVM': 300,
    'DBF': 600,
    'DNF': 600,
    'EBF': 700,
    'ENF': 600,
    'GRA': 150,
    'MF':  600,
    'OSH': 200,
    'SAV': 400,
    'SNO': 500,
    'URB': 500,
    'WAT': 300,
    'WET': 250,
    'WSA': 400,
}

def harmonise_ETM(img):
    B3_harmonised = img.select('SR_B3') \
        .multiply(0.9047).add(0.0061).rename('SR_B3_har')
    B4_harmonised = img.select('SR_B4') \
        .multiply(0.8462).add(0.0412).rename('SR_B4_har')
    return img.addBands([B3_harmonised, B4_harmonised])

def radiometric_calibration(img, input_band = 'R', output_band='R_cal'):
    band = img.select(input_band)
    band_cal = band.divide(10000).rename(output_band)
    return img.addBands(band_cal)

def add_ndvi(img, red_band='R', nir_band='NIR', output_band='NDVI'):
    red = img.select(red_band)
    nir = img.select(nir_band)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename(output_band)
    return img.addBands(ndvi)

def add_nirv(img, red_band='R', nir_band='NIR', output_band='NIRv'):
    red = img.select(red_band)
    nir = img.select(nir_band)
    ndvi = nir.subtract(red).divide(nir.add(red))
    nirv = ndvi.multiply(nir).rename(output_band)
    return img.addBands(nirv)

def add_kndvi(img, red_band='R', nir_band='NIR', output_band='kNDVI'):
    red = img.select(red_band)
    nir = img.select(nir_band)
    ndvi = nir.subtract(red).divide(nir.add(red))
    kndvi = ndvi.pow(2).tanh().rename(output_band)
    return img.addBands(kndvi)

def add_evi2(img, red_band='R', nir_band='NIR', output_band='EVI2'):
    red = img.select(red_band)
    nir = img.select(nir_band)
    evi2 = nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(2.4)).add(1)).rename(output_band)
    return img.addBands(evi2)