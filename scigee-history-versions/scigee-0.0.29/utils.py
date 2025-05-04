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