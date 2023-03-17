import numpy as np
import ee
import math
import datetime


irSumThresh = 0.3
ndviWaterThresh = -0.1
erodePixels = 1.5
dilationPixels = 3
SR_BAND_SCALE = 1e4
cloudFreeKeepThresh = 1
BUCKET = "lyy-shafts"
DATA_FOLDER = "dataset_tmp"


def computeQualityScore(img: ee.Image):
  # ------QualityScore is calculated by selecting the maximum value between the Cloud Score and Shadow Score for each pixel
  score = img.select(['cloudScore']).max(img.select(['shadowScore']))

  score = score.reproject('EPSG:4326', None, 20).reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(5))

  score = score.multiply(-1)

  return img.addBands(score.rename('cloudShadowScore'))


def dilatedErossion(score):
  # Perform opening on the cloud scores
  score = score.reproject('EPSG:4326', None, 20) \
            .focalMin(radius=erodePixels, kernelType='circle', iterations=3) \
            .focalMin(radius=dilationPixels, kernelType='circle', iterations=3) \
            .reproject('EPSG:4326', None, 20)

  return score


def mergeCollection_LightGBM(imgC: ee.ImageCollection):
  # Select the best images, which are below the cloud free threshold, sort them in reverse order (worst on top) for mosaicing
  best = imgC.filter(ee.Filter.lt('CLOUDY_PERCENTAGE', cloudFreeKeepThresh)).sort('CLOUDY_PERCENTAGE_ROI', False)
  
  # Composites all the images in a collection, using a quality band as a per-pixel ordering function (use pixels with the HIGHEST score).
  filtered = imgC.qualityMosaic('cloudShadowScore')

  # Add the quality mosaic to fill in any missing areas of the ROI which aren't covered by good images
  newC = ee.ImageCollection.fromImages([filtered, best.mosaic()])

  # Note that the `mosaic` method composites overlapping images according to their order in the collection (last, i.e., best, on top)
  return ee.Image(newC.mosaic())
  
  #return best.median()


def add_cloud_bands(img: ee.Image, cloud_prob_threshold=50):
  cloud_prob = ee.Image(img.get("s2cloudless")).select("probability").divide(100).rename("cloudScore")
  # ------1 for cloudy pixels and 0 for non-cloudy pixels
  is_cloud = cloud_prob.gt(cloud_prob_threshold / 100.0).rename("cloudFlag")

  return img.addBands(ee.Image([cloud_prob, is_cloud]))


def add_shadow_bands(img: ee.Image):
  meanAzimuth = img.get('MEAN_SOLAR_AZIMUTH_ANGLE')
  meanZenith = img.get('MEAN_SOLAR_ZENITH_ANGLE')

  cloudMask = img.select(['cloudFlag'])

  #Find dark pixels
  darkPixelsImg = img.select(['B8','B11','B12']).divide(SR_BAND_SCALE).reduce(ee.Reducer.sum())
  # darkPixelsImg = img.select(['B1','B11','B12']).divide(SR_BAND_SCALE).reduce(ee.Reducer.sum())

  ndvi = img.normalizedDifference(['B8','B4'])
  waterMask = ndvi.lt(ndviWaterThresh)
  darkPixels = darkPixelsImg.lt(irSumThresh)

  # Get the mask of pixels which might be shadows excluding water
  darkPixelMask = darkPixels.And(waterMask.Not())
  darkPixelMask = darkPixelMask.And(cloudMask.Not())

  #Find where cloud shadows should be based on solar geometry
  #Convert to radians
  azR = ee.Number(meanAzimuth).add(180).multiply(math.pi).divide(180.0)
  zenR = ee.Number(meanZenith).multiply(math.pi).divide(180.0)

  #Find the shadows
  def func_uke(cloudHeight):
    cloudHeight = ee.Number(cloudHeight)

    shadowCastedDistance = zenR.tan().multiply(cloudHeight) #Distance shadow is cast
    x = azR.sin().multiply(shadowCastedDistance).multiply(-1) #.divide(nominalScale);#X distance of shadow
    y = azR.cos().multiply(shadowCastedDistance).multiply(-1) #Y distance of shadow
    # ------`displace` warps an image using an image of displacements.
    return img.select(['cloudScore']).displace(ee.Image.constant(x).addBands(ee.Image.constant(y)))

  cloudHeights = ee.List.sequence(200, 10000, 250)
  shadows = cloudHeights.map(func_uke)
  shadowMasks = ee.ImageCollection.fromImages(shadows)
  shadowMask = shadowMasks.mean()

  # #Create shadow mask
  shadowMask = dilatedErossion(shadowMask.multiply(darkPixelMask))

  shadowScore = shadowMask.reduceNeighborhood(reducer=ee.Reducer.max(), kernel=ee.Kernel.square(1))

  img = img.addBands(shadowScore.rename(['shadowScore']))

  return img


def add_cloud_shadow_mask(img: ee.Image, cloud_prob_threshold=50):
  img_cloud = add_cloud_bands(img, cloud_prob_threshold)
  img_cloud_shadow = add_shadow_bands(img_cloud)

  imgPoly = ee.Algorithms.GeometryConstructors.Polygon(
            ee.Geometry(img.get("system:footprint")).coordinates()
  )
  roi = ee.Geometry(img.get('ROI'))
  intersection = roi.intersection(imgPoly, ee.ErrorMargin(0.5))
  cloudMask = img_cloud_shadow.select("cloudFlag").clip(roi).rename("cloudMask")

  cloudAreaImg = cloudMask.multiply(ee.Image.pixelArea())
  stats = cloudAreaImg.reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e14)

  # ------area of cloudy pixels / area of Sentinel-2 images
  cloudPercent = ee.Number(stats.get('cloudMask')).divide(imgPoly.area()).multiply(100)

  # ------area of intersection with the city boundary / whole area of the city boundary
  coveragePercent = ee.Number(intersection.area()).divide(roi.area()).multiply(100)
  # ------area of cloudy pixels / area of the city boundary
  cloudPercentROI = ee.Number(stats.get('cloudMask')).divide(roi.area()).multiply(100)

  img_cloud_shadow = img_cloud_shadow.set('CLOUDY_PERCENTAGE', cloudPercent)
  img_cloud_shadow = img_cloud_shadow.set('ROI_COVERAGE_PERCENT', coveragePercent)
  img_cloud_shadow = img_cloud_shadow.set('CLOUDY_PERCENTAGE_ROI', cloudPercentROI)

  return img_cloud_shadow


lon_min = -0.50
lat_min = 51.00
lon_max = -0.45
lat_max = 51.05
year = 2020
dx = 0.09
dy = 0.09
padding = 0.02
target_resolution = 100
patch_size_ratio = 1
s2_cloud_prob_threshold = 20
s2_cloud_prob_max = 80


point_left_top = [lon_min, lat_max]
point_right_top = [lon_max, lat_max]
point_right_bottom = [lon_max, lat_min]
point_left_bottom = [lon_min, lat_min]

dx = padding
dy = padding
point_left_top = (point_left_top[0]-dx, point_left_top[1]+dy)
point_right_top = (point_right_top[0]+dx, point_right_top[1]+dy)
point_right_bottom = (point_right_bottom[0]+dx, point_right_bottom[1]-dy)
point_left_bottom = (point_left_bottom[0]-dx, point_left_bottom[1]-dy)
target_bd = ee.Geometry.Polygon([point_left_top, point_right_top, point_right_bottom, point_left_bottom, point_left_top])

# ------set the sample point
spc = 0.0009
xloc = np.arange(lon_min + spc * 0.5, lon_max, spc)
yloc = np.arange(lat_max - spc * 0.5, lat_min, -spc)
x, y = np.meshgrid(xloc, yloc)
xy_coords = np.concatenate([x.flatten().reshape(-1, 1), y.flatten().reshape(-1, 1)], axis=1)
target_pt = ee.FeatureCollection([ee.Geometry.Point(cr[0], cr[1]) for cr in xy_coords])

# ------set the name of the exported TFRecords file
records_name = "_" + "H{0}".format(len(yloc)) + "W{0}".format(len(xloc))

# ------filter the NASADEM data
DEM_dta = ee.Image("NASA/NASADEM_HGT/001").select(["elevation"]).float()
DEM_dta = DEM_dta.clip(target_bd)

# ------filter the Sentinel-1 data
s1_ds = ee.ImageCollection("COPERNICUS/S1_GRD")
s1_ds = s1_ds.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
s1_ds = s1_ds.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
s1_ds = s1_ds.filter(ee.Filter.eq('instrumentMode', 'IW'))
s1_ds = s1_ds.select(["VV", "VH"])

s1_year = min(2022, max(year, 2015))    # Sentinel-1 availability: from 2014-10 to 2022-12
if 0.5 * (lat_max + lat_min) > 0:
    s1_time_start = datetime.datetime(year=s1_year, month=12, day=1).strftime("%Y-%m-%d")
    s1_time_end = datetime.datetime(year=s1_year+1, month=3, day=1).strftime("%Y-%m-%d")
else:
    s1_time_start = datetime.datetime(year=s1_year, month=6, day=1).strftime("%Y-%m-%d")
    s1_time_end = datetime.datetime(year=s1_year, month=9, day=1).strftime("%Y-%m-%d")
s1_ds = s1_ds.filter(ee.Filter.date(s1_time_start, s1_time_end))
s1_ds = s1_ds.filterBounds(target_bd)
s1_img = s1_ds.reduce(ee.Reducer.percentile([50]))
s1_img = s1_img.clip(target_bd)
# ------transform VV/VH bands of Sentinel-1 images to corresponding backscattering coefficients for the compatibility with SHAFTS
# s1_img = get_backscatterCoef_EE(s1_img)

# ------filter the Sentinel-2 cloud-free data
s2_cld_threshold_base = 50
s2_cld_prob_step = 5
s2_year = min(2022, max(year, 2018))    # Sentinel-2 availability: from 2017-03 to 2022-12

s2_season = {"spring": ["{0}-03-01".format(s2_year), "{0}-06-01".format(s2_year)],
                "summer": ["{0}-06-01".format(s2_year), "{0}-09-01".format(s2_year)],
                "autumn": ["{0}-09-01".format(s2_year), "{0}-12-01".format(s2_year)],
                "winter": ["{0}-12-01".format(s2_year), '{0}-03-01'.format(s2_year+1)],
}
if 0.5 * (lat_max + lat_min) > 0:
    period_list = ["autumn", "spring", "summer", "winter"]
else:
    period_list = ["spring", "autumn", "winter", "summer"]

s2_img_found_flag = False
s2_img_cloudFree = None
for period in period_list:
    if not s2_img_found_flag:
        s2_time_start_str, s2_time_end_str = s2_season[period]
        s2_time_start = datetime.datetime.strptime(s2_time_start_str, "%Y-%m-%d")
        s2_time_end = datetime.datetime.strptime(s2_time_end_str, "%Y-%m-%d")

        s2_sr_col = ee.ImageCollection("COPERNICUS/S2_SR") \
                .filterBounds(target_bd) \
                .filter(ee.Filter.date(s2_time_start, s2_time_end)) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', s2_cld_threshold_base))
    
        s2_cloudless_col = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY") \
                            .filterBounds(target_bd) \
                            .filter(ee.Filter.date(s2_time_start, s2_time_end))
        
        s2_combined_col = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
                'primary': s2_sr_col,
                'secondary': s2_cloudless_col,
                'condition': ee.Filter.equals(**{
                    'leftField': 'system:index',
                    'rightField': 'system:index'
                })
        }))
        s2_combined_col = s2_combined_col.map(lambda x: x.clip(target_bd)).map(lambda x: x.set("ROI", target_bd))

        cld_prb_set = np.arange(s2_cloud_prob_threshold, s2_cloud_prob_max + s2_cld_prob_step, s2_cld_prob_step)
        num_cld_prb = len(cld_prb_set)

        for cld_prb_id in range(0, num_cld_prb):
            if not s2_img_found_flag:
                cld_prb_v = int(cld_prb_set[cld_prb_id])
                s2_combined_col_tmp = s2_combined_col.map(lambda x: add_cloud_shadow_mask(x, cld_prb_v)).map(computeQualityScore).sort("CLOUDY_PERCENTAGE")
                s2_img_cloudFree = mergeCollection_LightGBM(s2_combined_col_tmp)
                s2_img_cloudFree = s2_img_cloudFree.reproject('EPSG:4326', None, 10)
                s2_img_cloudFree = s2_img_cloudFree.clip(target_bd)
                #s2_img_found_flag = True
                bandnamelist = s2_img_cloudFree.bandNames().getInfo()
                if len(bandnamelist) > 0:
                    s2_img_found_flag = True
            else:
                break
    else:
        break

s2_img_cloudFree = s2_img_cloudFree.select(["B4", "B3", "B2", "B8"])
# ------rescale R/G/B/NIR bands of Sentinel-2 to [0, 255] for the compatibility with SHAFTS
#s2_img_cloudFree = get_normalizedImage_EE(s2_img_cloudFree)

kernelSize = int(target_resolution / 10.0)
overlapSize = 10
sentinel_image = ee.Image.cat([s1_img, s2_img_cloudFree]).float()

# ---------generate the neighborhood patches from Sentinel imagery
sentinel_psize = kernelSize + overlapSize
sentinel_value_default = ee.List.repeat(-1e4, sentinel_psize)
sentinel_value_default = ee.List.repeat(sentinel_value_default, sentinel_psize)
sentinel_kernel = ee.Kernel.fixed(sentinel_psize, sentinel_psize, sentinel_value_default)

sentinel_patched = sentinel_image.neighborhoodToArray(sentinel_kernel)
target_pt = sentinel_patched.sampleRegions(collection=target_pt, scale=10, geometries=True)

# ---------generate the neighborhood patches from DEM data
dem_psize = int(sentinel_psize * patch_size_ratio)
dem_value_default = ee.List.repeat(-1e4, dem_psize)
dem_value_default = ee.List.repeat(dem_value_default, dem_psize)
dem_kernel = ee.Kernel.fixed(dem_psize, dem_psize, dem_value_default)

dem_patched = DEM_dta.neighborhoodToArray(dem_kernel)
target_pt = dem_patched.sampleRegions(collection=target_pt, scale=30)

# ---------export center points for sampling patches
'''
shp_task_config = {
    "collection": target_pt,
    "description": records_name,
    "folder": dst_dir,
    "fileFormat": "SHP",
}

shp_task = ee.batch.Export.table.toDrive(**shp_task_config)
shp_task.start()
'''

'''
# ---------export the image where patches are sampled
s1_img_task_config = {
    "image": s1_img.float(),
    "description": "S1" + records_name,
    "folder": dst_dir,
    "scale": 10,
    "maxPixels": 1e13,
    "crs": "EPSG:4326"
}
s1_img_task = ee.batch.Export.image.toDrive(**s1_img_task_config)
s1_img_task.start()

s2_img_task_config = {
    "image": s2_img_cloudFree.float(),
    "description": "S2" + records_name,
    "folder": dst_dir,
    "scale": 10,
    "maxPixels": 1e13,
    "crs": "EPSG:4326"
}
s2_img_task = ee.batch.Export.image.toDrive(**s2_img_task_config)
s2_img_task.start()

nasaDEM_img_task_config = {
    "image": DEM_dta.float(),
    "description": "nasaDEM" + records_name,
    "folder": dst_dir,
    "scale": 30,
    "maxPixels": 1e13,
    "crs": "EPSG:4326"
}
nasaDEM_img_task = ee.batch.Export.image.toDrive(**nasaDEM_img_task_config)
nasaDEM_img_task.start()
'''

# ---------export the patched dataset into the format of TFRecord

records_task_config = {
    "collection": target_pt,
    "description": records_name,
    "bucket": BUCKET,
    "fileNamePrefix": DATA_FOLDER + "/" + records_name,
    "fileFormat": "TFRecord",
}
records_task = ee.batch.Export.table.toCloudStorage(**records_task_config)
records_task.start()