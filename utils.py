import os
import argparse
import math
import numpy as np
import datetime
import pandas as pd
from osgeo import gdal, ogr, osr
import ee


ee.Initialize()

cloudFreeKeepThresh = 1

# ********* Parameters for shadow detection *********
cloudHeights = ee.List.sequence(200, 10000, 250)
irSumThresh = 0.3
ndviWaterThresh = -0.1
erodePixels = 1.5
dilationPixels = 3

# ********* Parameters for s2cloudless *********
S2_YEAR_MAX = 2022
SR_BAND_SCALE = 1e4
# BUFFER_SIZE = 30
# CLOUD_PROB_THRESHOLD = 60
CLOUD_PROB_THRESHOLD_step = 5
CLOUD_PROB_THRESHOLD_max = 75
# NIR_DARK_THRESHOLD = 0.15
NIR_DARK_THRESHOLD_step = 0.02
NIR_DARK_THRESHOLD_min = 0.07
# CLD_PRJ_DIST = 1
CLD_PRJ_DIST_step = 0.5
CLD_PRJ_DIST_min = 1


def get_backscatterCoef_EE(raw_s1_img: ee.Image):
  img_10 = ee.Image(10)
  coef = img_10.pow(raw_s1_img.divide(img_10)).clamp(0.0, 1.0)

  return coef


def get_normalizedImage_EE(img: ee.Image, scale=10, q1=0.98, q2=0.02, vmin=0.0, vmax=1.0):
  bandnamelist = img.bandNames().getInfo()
  roi = img.geometry()
  band_normalizedList = []
  q1_100 = int(q1 * 100)
  q2_100 = int(q2 * 100)
  for b in bandnamelist:
    min_max_ref = img.select(b).reduceRegion(reducer=ee.Reducer.percentile([q1_100, q2_100]), geometry=roi, scale=scale, maxPixels=1e14)
    print(min_max_ref.getInfo())
    band_n = img.select(b).unitScale(min_max_ref.get(b+"_p{0}".format(q2_100)), min_max_ref.get(b+"_p{0}".format(q1_100)))
    band_normalizedList.append(band_n)

  img_n = ee.Image.cat(band_normalizedList).clamp(vmin, vmax)

  return img_n


def getPopInfo(dx=0.09, dy=0.09):
  # ------ref to: https://unstats.un.org/unsd/demographic/sconcerns/densurb/defintion_of%20urban.pdf
  # ------ref to: https://ourworldindata.org/urbanization
  hrsl_popC = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop")

  lon_min_list = []
  lon_max_list = []
  lat_min_list = []
  lat_max_list = []
  area_list = []
  num_pop_list = []
  pop_density_list = []
  for lon_min in np.arange(-180, 180, dx):
    lon_max = lon_min + dx
    for lat_min in np.arange(-90, 90, dy):
      lat_max = lat_min + dy

      point_left_top = [lon_min, lat_max]
      point_right_top = [lon_max, lat_max]
      point_right_bottom = [lon_max, lat_min]
      point_left_bottom = [lon_min, lat_min]

      target_bd = ee.Geometry.Polygon([point_left_top, point_right_top, point_right_bottom, point_left_bottom, point_left_top])
      hrsl_popC_tmp = hrsl_popC.filterBounds(target_bd)
      hrsl_pop_img = hrsl_popC_tmp.mosaic()
      hrsl_pop_img = hrsl_pop_img.clip(target_bd)
      # ------calculate the area of the target region (in km2)
      hrsl_area = ee.Number(hrsl_pop_img.geometry().area())
      hrsl_area = hrsl_area.getInfo() / 1000.0 / 1000.0
      # ------calculate the neibhboring population density for the target region
      # ------calculate the total number of population for the target region
      pop_sum = hrsl_pop_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=target_bd, scale=30, maxPixels=1e14)
      pop_sum = pop_sum.get('b1').getInfo()
      # print(hrsl_area.getInfo(), "\t", pop_sum.get('b1').getInfo())
      lon_min_list.append(lon_min)
      lon_max_list.append(lon_max)
      lat_min_list.append(lat_min)
      lat_max_list.append(lat_max)
      area_list.append(hrsl_area)
      num_pop_list.append(pop_sum)
      pop_density_list.append(pop_sum / hrsl_area)
      print("\t".join([str(lon_min), str(lon_max), str(lat_min), str(lat_max), str(pop_sum / hrsl_area)]))
  
  hrsl_df = pd.DataFrame({"lon_min": lon_min_list, "lon_max": lon_max_list, "lat_min": lat_min_list, "lat_max": lat_max_list, "area": area_list, "num_pop": num_pop_list, "pop_density": pop_density_list})
  hrsl_df.to_csv("HRSL_info.csv", index=False)
  
  '''
  task_config = {"image": hrsl_pop_img.toFloat(),
                  "description": "hrsl_tmp",
                  "folder": "Global_export",
                  "scale": 30,
                  "maxPixels": 1e13,
                  "crs": 'EPSG:4326'}
  task = ee.batch.Export.image.toDrive(**task_config)
  
  task.start()
  '''


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


def apply_cloud_shadow_mask(img: ee.Image):
  # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
  not_cld_shdw = img.select('cloudMask').Not()

  # Subset reflectance bands and update their masks, return the result.
  return img.updateMask(not_cld_shdw)


def add_cloud_bands(img: ee.Image, cloud_prob_threshold=50):
  cloud_prob = ee.Image(img.get("s2cloudless")).select("probability").divide(100).rename("cloudScore")
  # ------1 for cloudy pixels and 0 for non-cloudy pixels
  is_cloud = cloud_prob.gt(cloud_prob_threshold / 100.0).rename("cloudFlag")

  return img.addBands(ee.Image([cloud_prob, is_cloud]))


def dilatedErossion(score):
  # Perform opening on the cloud scores
  score = score.reproject('EPSG:4326', None, 20) \
            .focalMin(radius=erodePixels, kernelType='circle', iterations=3) \
            .focalMin(radius=dilationPixels, kernelType='circle', iterations=3) \
            .reproject('EPSG:4326', None, 20)

  return score


def computeQualityScore(img: ee.Image):
  # ------QualityScore is calculated by selecting the maximum value between the Cloud Score and Shadow Score for each pixel
  score = img.select(['cloudScore']).max(img.select(['shadowScore']))

  score = score.reproject('EPSG:4326', None, 20).reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(5))

  score = score.multiply(-1)

  return img.addBands(score.rename('cloudShadowScore'))


# def add_shadow_bands(img: ee.Image, nir_dark_threshold=0.15, cloud_proj_distance=1):
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

  shadows = cloudHeights.map(func_uke)
  shadowMasks = ee.ImageCollection.fromImages(shadows)
  shadowMask = shadowMasks.mean()

  # #Create shadow mask
  shadowMask = dilatedErossion(shadowMask.multiply(darkPixelMask))

  shadowScore = shadowMask.reduceNeighborhood(reducer=ee.Reducer.max(), kernel=ee.Kernel.square(1))

  img = img.addBands(shadowScore.rename(['shadowScore']))

  return img

'''
def add_shadow_bands(img: ee.Image, nir_dark_threshold=0.15, cloud_proj_distance=1):
  # ------identify the water pixels from the SCL band
  not_water = img.select("SCL").neq(6)

  # ------identify dark NIR pixels that are not water (potential cloud shadow pixels)
  dark_pixels = img.select("B8").lt(nir_dark_threshold * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

  # ------determine the direction to project cloud shadow from clouds
  shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

  # ------project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
  # ---------`directionalDistanceTransform` function is used to calculate the distance between 
  # ---------the nearest cloudy pixels and current non-cloudy pixels along `shadow_azimuth` (<= CLD_PRJ_DIST*10)
  cld_proj = img.select('cloudFlag').directionalDistanceTransform(shadow_azimuth, cloud_proj_distance * 10) \
              .reproject(**{'crs': img.select(0).projection(), 'scale': 100}) \
              .select('distance') \
              .mask() \
              .rename('cloud_transform')

  # ------identify the intersection of dark pixels with cloud shadow projection.
  shadows = cld_proj.multiply(dark_pixels).rename('shadowFlag')

  # ------add dark pixels, cloud projection, and identified shadows as image bands.
  img = img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

  score = img.select(['cloudFlag']).max(img.select(['shadowFlag']))
  score = score.reproject('EPSG:4326', None, 20).reduceNeighborhood(reducer=ee.Reducer.max(), kernel=ee.Kernel.square(3))
  score = score.multiply(-1)
  
  return img.addBands(score.rename('cloudShadowScore'))
'''


# def add_cloud_shadow_mask(img: ee.Image, bf_size=3, cloud_prob_threshold=50, nir_dark_threshold=0.15, cloud_proj_distance=1):
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


# ---ref to: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
# def exportCloudFreeSen2_LightGBM(file_name, dst_dir, date_interval, roi, bf_size=3, cloud_prob_threshold=20, nir_dark_threshold=0.15, cloud_proj_distance=2, dst="Drive"):
def exportCloudFreeSen2_LightGBM(file_name, dst_dir, date_interval, roi, cloud_prob_threshold=40, dst="Drive"):
  cld_threshold_base = 50
  time_start_str, time_end_str = date_interval

  target_year = int(time_start_str[0:4])

  # ------if no image can be found under the current criteria, we can try to relax:
  # ---------(0) increase the target year
  num_target_year = S2_YEAR_MAX - target_year + 1
  # ---------(1) increase the probability threshold for cloudy pixels
  # ---------(2) decrease the NIR threshold for dark pixels
  # ---------(3) decrease the distance for shadow pixel detection
  cld_prb_set = np.arange(cloud_prob_threshold, CLOUD_PROB_THRESHOLD_max + CLOUD_PROB_THRESHOLD_step, CLOUD_PROB_THRESHOLD_step)
  num_cld_prb = len(cld_prb_set)
  '''
  nir_dark_set = np.arange(nir_dark_threshold, NIR_DARK_THRESHOLD_min-NIR_DARK_THRESHOLD_step, -NIR_DARK_THRESHOLD_step)
  num_nir_dark = len(nir_dark_set)
  cld_prj_set = np.arange(cloud_proj_distance, CLD_PRJ_DIST_min-CLD_PRJ_DIST_step, -CLD_PRJ_DIST_step)
  num_cld_prj = len(cld_prj_set)
  sd_sum_max = num_nir_dark + num_cld_prj - 2
  '''
  img_found_flag = False
  cloudFree_img = None
  for year_delta in range(0, num_target_year):
    if not img_found_flag:
      y_start = target_year + year_delta
      t_start_str = str(y_start) + time_start_str[4:]
      t_start = datetime.datetime.strptime(t_start_str, "%Y-%m-%d")
      y_end = int(time_end_str[0:4]) + year_delta
      t_end_str = str(y_end) + time_end_str[4:]
      t_end = datetime.datetime.strptime(t_end_str, "%Y-%m-%d")

      #Filter images of this period
      s2_sr_col = ee.ImageCollection("COPERNICUS/S2_SR") \
                  .filterBounds(roi) \
                  .filter(ee.Filter.date(t_start, t_end)) \
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cld_threshold_base))
      
      s2_cloudless_col = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY") \
                          .filterBounds(roi) \
                          .filter(ee.Filter.date(t_start, t_end))

      s2_combined_col = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_sr_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
      }))

      s2_combined_col = s2_combined_col.map(lambda x: x.clip(roi)) \
                                        .map(lambda x: x.set("ROI", roi))

      for cld_prb_id in range(0, num_cld_prb):
        if not img_found_flag:
          cld_prb_v = int(cld_prb_set[cld_prb_id])
          s2_combined_col_tmp = s2_combined_col.map(lambda x: add_cloud_shadow_mask(x, cld_prb_v)).map(computeQualityScore).sort("CLOUDY_PERCENTAGE")
          cloudFree_img = mergeCollection_LightGBM(s2_combined_col_tmp)
          # cloudFree_img = s2_combined_col_tmp.median()
          cloudFree_img = cloudFree_img.reproject('EPSG:4326', None, 10)
          cloudFree_img = cloudFree_img.clip(roi)

          bandnamelist = cloudFree_img.bandNames().getInfo()
          if len(bandnamelist) > 0:
            img_found_flag = True
          '''
          sd_sum = 0
          while sd_sum <= sd_sum_max and not img_found_flag:
            nd_id_min = max(0, sd_sum - num_cld_prj + 1)
            nd_id_max = min(sd_sum+1, num_nir_dark)
            for nir_dark_id in range(nd_id_min, nd_id_max):
              nir_dark_v = float(nir_dark_set[nir_dark_id])
              cld_prj_id = sd_sum - nir_dark_id
              cld_prj_v = float(cld_prj_set[cld_prj_id])

              s2_combined_col_tmp = s2_combined_col.map(lambda x: add_cloud_shadow_mask(x, bf_size, cld_prb_v, nir_dark_v, cld_prj_v)).map(computeQualityScore).sort("CLOUDY_PERCENTAGE")
              cloudFree_img = mergeCollection_LightGBM(s2_combined_col_tmp)
              # cloudFree_img = s2_combined_col_tmp.median()
              cloudFree_img = cloudFree_img.reproject('EPSG:4326', None, 10)
              cloudFree_img = cloudFree_img.clip(roi)

              bandnamelist = cloudFree_img.bandNames().getInfo()
              if len(bandnamelist) > 0:
                img_found_flag = True
                break
              sd_sum += 1
          '''
        else:
          break
    else:
      break
  
  if cloudFree_img is not None:
    # print("Time range: {0}-{1} ".format(t_start_str, t_end_str), "CLOUD_PROB_THRESHOLD: {0} ".format(cld_prb_v), "NIR_DARK_THRESHOLD: {0} ".format(nir_dark_v), "CLD_PRJ_DIST: {0}".format(cld_prj_v))
    print("Time range: {0}-{1} ".format(t_start_str, t_end_str), "CLOUD_PROB_THRESHOLD: {0} ".format(cld_prb_v))
    
    # Export the various resolutions to the correct image folders
    # Assumes the GCS bucket is called sentinel2
    # Exports to folder structure "city/season/city_season_resolution.tiff"
    if dst == "Drive":
      task_config = {
        'image': cloudFree_img.select(["B4", "B3", "B2", "B8"]),
        'description': file_name,
        'scale': 10,
        'folder': dst_dir,
        'region': roi,
        'maxPixels': 1e13
      }
      task = ee.batch.Export.image.toDrive(**task_config)
    elif dst == "CloudStorage":
      task_config = {
        'image': cloudFree_img.select(["B4", "B3", "B2", "B8"]),
        'description': file_name,
        'scale': 10,
        'bucket': dst_dir,
        'region': roi,
        'maxPixels': 1e13
      }
      task = ee.batch.Export.image.toDrive(**task_config)

    task.start()
  else:
    print("Fail to find cloud-free image under current settings!!!")


# downloadSeasonsSen2: Exports all GeoTIFFs for all seasons for specific ROI
#        input: roi - Region of interest to clip, for exporting
#               roiID - A unique identifier for the ROI, used for naming files
#        output: None
def sentinel2cloudFree_download_by_extent(lon_min: float, lat_min: float, lon_max: float, lat_max: float, year: int, dst_dir: str, file_name: str, padding=0.04, cloud_prob_threshold=40, dst="Drive"):
  point_left_top = [lon_min, lat_max]
  point_right_top = [lon_max, lat_max]
  point_right_bottom = [lon_max, lat_min]
  point_left_bottom = [lon_min, lat_min]

  ee.Initialize()

  if padding is not None:
    if isinstance(padding, list):
        dx = padding[0]
        dy = padding[1]
    else:
        dx = padding
        dy = padding
    point_left_top = (point_left_top[0]-dx, point_left_top[1]+dy)
    point_right_top = (point_right_top[0]+dx, point_right_top[1]+dy)
    point_right_bottom = (point_right_bottom[0]+dx, point_right_bottom[1]-dy)
    point_left_bottom = (point_left_bottom[0]-dx, point_left_bottom[1]-dy)

  city_bd = ee.Geometry.Polygon([point_left_top, point_right_top, point_right_bottom, point_left_bottom, point_left_top])

  # DateRange is start inclusive and end exclusive
  seasons = {"spring": ["{0}-03-01".format(year), "{0}-06-01".format(year)],
                "summer": ["{0}-06-01".format(year), "{0}-09-01".format(year)],
                "autumn": ["{0}-09-01".format(year), "{0}-12-01".format(year)],
                "winter": ["{0}-12-01".format(year), '{0}-03-01'.format(year+1)],
  }

  for season in seasons:
    dates = seasons[season]
    # exportCloudFreeSen2(file_name+"_"+season, dst_dir, dates, city_bd, dst)
    exportCloudFreeSen2_LightGBM(file_name+"_"+season, dst_dir, dates, city_bd, cloud_prob_threshold, dst)


def sentinel2cloudFree_download(sample_csv: str, dst_dir: str, path_prefix=None, padding=0.04, target_epsg=4326, cloud_prob_threshold=40, dst="Drive"):
  df = pd.read_csv(sample_csv)
  target_spatialRef = osr.SpatialReference()
  target_spatialRef.ImportFromEPSG(target_epsg)
  # ------For GDAL 3.0, we must add the folllowing line, see the discussion:
  # ---------https://gis.stackexchange.com/questions/364943/gdal-3-0-4-invalid-coordinate-transformation-result
  target_spatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

  for row_id in df.index:
      # ------read basic information of Shapefile
      year = df.loc[row_id]["Year"]
      year = min(2022, max(year, 2018))   # Sentinel-2 availability: from 2017-03 to 2021-12
      dta_path = os.path.join(str(path_prefix or ''), df.loc[row_id]["Path"])

      # ------get GeoTiff extent under the target projection (Default: WGS84, EPSG:4326)
      # ---------if the data file is ESRI Shapefile
      if dta_path.endswith(".shp") or dta_path.endswith(".gpkg"):
          shp_ds = ogr.Open(dta_path, 0)
          shp_layer = shp_ds.GetLayer()
          shp_spatialRef = shp_layer.GetSpatialRef()
          coordTrans = osr.CoordinateTransformation(shp_spatialRef, target_spatialRef)
          x_min, x_max, y_min, y_max = shp_layer.GetExtent()
      # ---------otherwise, it must be GeoTiff file
      else:
          tiff_ds = gdal.Open(dta_path, 0)
          tiff_spatialRef = osr.SpatialReference(wkt=tiff_ds.GetProjection())
          coordTrans = osr.CoordinateTransformation(tiff_spatialRef, target_spatialRef)
          tiff_geoTransform = tiff_ds.GetGeoTransform()
          x_min = tiff_geoTransform[0]
          y_max = tiff_geoTransform[3]
          x_max = x_min + tiff_geoTransform[1] * tiff_ds.RasterXSize
          y_min = y_max + tiff_geoTransform[5] * tiff_ds.RasterYSize

      point_left_top = coordTrans.TransformPoint(x_min, y_max)[0:2]
      point_left_bottom = coordTrans.TransformPoint(x_min, y_min)[0:2]
      point_right_top = coordTrans.TransformPoint(x_max, y_max)[0:2]
      point_right_bottom = coordTrans.TransformPoint(x_max, y_min)[0:2]
      
      x_min = min(point_left_top[0], point_left_bottom[0], point_right_top[0], point_right_bottom[0])
      x_max = max(point_left_top[0], point_left_bottom[0], point_right_top[0], point_right_bottom[0])
      y_min = min(point_left_top[1], point_left_bottom[1], point_right_top[1], point_right_bottom[1])
      y_max = max(point_left_top[1], point_left_bottom[1], point_right_top[1], point_right_bottom[1])
      
      file_prefix = df.loc[row_id]["City"] + "_" + str(year)
      file_name = '{0}_sentinel_2_50pt'.format(file_prefix)

      sentinel2cloudFree_download_by_extent(lon_min=x_min, lat_min=y_min, lon_max=x_max, lat_max=y_max, year=year, dst_dir=dst_dir, file_name=file_name, 
                                              padding=padding, cloud_prob_threshold=cloud_prob_threshold, dst=dst)
      
      print(df.loc[row_id]["City"], "Start.")


if __name__ == "__main__":
  '''
  sentinel2cloudFree_download(sample_csv="GEE_Download_2022_back.csv", dst_dir="Sentinel-2_export_CF", path_prefix="/Volumes/ForLyy/Temp/ReferenceData", 
                                padding=0.05, cloud_prob_threshold=30, dst="Drive")
  '''
  getPopInfo(dx=0.09, dy=0.09)