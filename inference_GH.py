from typing import Union, List, Dict
import argparse
import numpy as np
import datetime
import time
import ee

from googleapiclient import discovery
from googleapiclient.errors import HttpError
from google.cloud import storage
from google.oauth2 import service_account

# from utils import *


# FEATURES = ["VV_p50", "VH_p50", "B4", "B3", "B2", "B8", "elevation"]
FEATURES = ["VV", "VH", "B4", "B3", "B2", "B8", "elevation"]

kernel_size_ref = {100: 20, 250: 40, 500: 80, 1000: 160}
overlapSize_ref = {100: 10, 250: 15, 500: 30, 1000: 60}
degree_ref = {100: 0.0009, 250: 0.00225, 500: 0.0045, 1000: 0.009}

cloudFreeKeepThresh = 2
cloudScale = 30

# ********* Parameters for shadow detection *********
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


def GCS_setting(credentials_path: str, project_name="3DBuildingMapper", bucket_name="3DBuildingMapper", gee_data_folder="GEE_dataset_tmp"):
    # ---create a Google Cloud Project
    PROJECT_ID = "202303281215"
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    gcs_manager = discovery.build('cloudresourcemanager', 'v1', cache_discovery=False, credentials=credentials)
    gcp_body = {"projectID": PROJECT_ID, "name": project_name}
    request = gcs_manager.projects().create(body=gcp_body)
    request.execute()
    # ------enable the google earth engine API
    ee_request = gcs_manager.services().enable(name=f'projects/{0}/services/{1}'.format(PROJECT_ID, "earthengine.googleapis.com"))
    ee_request.execute()
    
    # ---set up the Google Cloud Storage
    client = storage.Client.from_service_account_json(credentials_path)
    # ------create a bucket
    bucket = client.create_bucket(bucket_name)
    # ------create a folder under the current bucket
    blob = storage.Blob("{0}/".format(gee_data_folder), bucket)
    blob.upload_from_string("")


def computeQualityScore_infer(img: ee.Image):
  # ------QualityScore is calculated by selecting the maximum value between the Cloud Score and Shadow Score for each pixel
  score = img.select(['cloudScore']).max(img.select(['shadowFlag']))

  score = score.reproject('EPSG:4326', None, 30).reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(1))

  score = score.multiply(-1)

  return img.addBands(score.rename('cloudShadowScore'))


def add_cloud_bands_infer(img: ee.Image, cloud_prob_threshold=50):
  cloud_prob = ee.Image(img.get("s2cloudless")).select("probability").rename("cloudScore")
  # ------1 for cloudy pixels and 0 for non-cloudy pixels
  is_cloud = cloud_prob.gt(cloud_prob_threshold).rename("cloudFlag")

  return img.addBands(ee.Image([cloud_prob, is_cloud]))


# ------recommended for exporting cloud-free Sentinel-2's images during inference
def add_shadow_bands_infer(img: ee.Image, nir_dark_threshold=0.15, cloud_proj_distance=1.5):
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
  img = img.addBands(ee.Image([cld_proj, shadows]))

  return img


def add_cloud_shadow_mask_infer(img: ee.Image, cloud_prob_threshold=50):
  img_cloud = add_cloud_bands_infer(img, cloud_prob_threshold)
  img_cloud_shadow = add_shadow_bands_infer(img_cloud)

  #roi = ee.Geometry(img.get('ROI'))
  imgPoly = ee.Algorithms.GeometryConstructors.Polygon(
            ee.Geometry(img.get("system:footprint")).coordinates()
  )

  cloudMask = img_cloud_shadow.select("cloudFlag").add(img_cloud_shadow.select("shadowFlag")).gt(0)
  cloudMask = (cloudMask.focalMin(1.5).focalMax(1.5).reproject(**{'crs': img.select([0]).projection(), 'scale': cloudScale}).rename('cloudmask'))
  cloudAreaImg = cloudMask.multiply(ee.Image.pixelArea())

  stats = cloudAreaImg.reduceRegion(reducer=ee.Reducer.sum(), scale=cloudScale, maxPixels=1e14)

  cloudPercent = ee.Number(stats.get('cloudmask')).divide(imgPoly.area()).multiply(100)  
  img_cloud_shadow = img_cloud_shadow.set('CLOUDY_PERCENTAGE', cloudPercent)
  #cloudPercentROI = ee.Number(stats.get('cloudmask')).divide(roi.area()).multiply(100)
  #img_cloud_shadow = img_cloud_shadow.set('CLOUDY_PERCENTAGE_ROI', cloudPercentROI)
  
  return img_cloud_shadow


def export_satData_GEE(DEM_dta: ee.Image, S1_dataset: ee.ImageCollection, S2_dataset: List[ee.ImageCollection], S2_cloudProb_dataset: ee.ImageCollection, lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], year: int, target_resolution: int, GCS_config: Dict[str, str], dst_dir: str, precision=2, destination="CloudStorage", file_prefix=None, padding=0.02, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80):
    """Export satellite images and data to Google Cloud Storage (GCS).

    Parameters
    ----------

    lon_min : float
        Minimum longitude of target region.
    lat_min : float
        Minimum latitude of target region.
    lon_max : float
        Maximum longitude of target region.
    lat_max : float
        Maximum latitude of target region.
    year : int
        Year of satellite images to be downloaded.
    dst_dir : str
        Directory on the destination device for saving the output image.
    file_prefix : str
        Name of the output image.
    padding : float
        Padding size outside the target region (in degrees).
        The default is `0.04`.

    """
    BUCKET = GCS_config["BUCKET"]
    DATA_FOLDER = GCS_config["DATA_FOLDER"]

    if any([not isinstance(lon_min, int), not isinstance(lon_max, int), not isinstance(lat_min, int), not isinstance(lat_max, int)]):
        lon_min_str = str(round(lon_min, precision))
        lat_min_str = str(round(lat_min, precision))
        lon_max_str = str(round(lon_max, precision))
        lat_max_str = str(round(lat_max, precision))
        file_suffix = "_".join([lon_min_str, lon_max_str, lat_min_str, lat_max_str])
    else:
        file_suffix = "_".join([str(lon_min), str(lon_max_str), str(lat_min_str), str(lat_max_str)])

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
    spc = degree_ref[target_resolution]
    xloc = ee.List.sequence(lon_min + spc * 0.5, lon_max, spc)
    yloc = ee.List.sequence(lat_max - spc * 0.5, lat_min, -spc)
    xy_coords = yloc.map(lambda y: xloc.map(lambda x: ee.Feature(ee.Geometry.Point([x, y])))).flatten()
    target_pt = ee.FeatureCollection(xy_coords)

    # ------set the name of the exported TFRecords file
    if file_prefix is not None:
        records_name = file_prefix + "_" + file_suffix + "_" + "H{0}".format(100) + "W{0}".format(100)
    else:
        records_name = "_" + file_suffix + "_" + "H{0}".format(100) + "W{0}".format(100)

    # ------filter the NASADEM data
    # DEM_dta = ee.Image("NASA/NASADEM_HGT/001").select(["elevation"]).float()
    DEM_dta = DEM_dta.clip(target_bd)
    
    # ------filter the Sentinel-1 data
    S1_dataset = S1_dataset.filterBounds(target_bd)
    # ------note that if we use difference aggregation ops, exported bands would have different suffix with "_p50"
    s1_img = S1_dataset.mean().clip(target_bd)

    # ------filter the Sentinel-2 cloud-free data
    s2_cloudless_col =  S2_cloudProb_dataset.filterBounds(target_bd)

    s2_cld_prob_step = 5
    cld_prb_set = np.arange(s2_cloud_prob_threshold, s2_cloud_prob_max + s2_cld_prob_step, s2_cld_prob_step)
    num_cld_prb = len(cld_prb_set)
    
    s2_img_found_flag = False
    s2_img_cloudFree = None
    for s2_ds_tmp in S2_dataset:
        if not s2_img_found_flag:
            s2_sr_col = s2_ds_tmp.filterBounds(target_bd)
            s2_combined_col = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
                    'primary': s2_sr_col,
                    'secondary': s2_cloudless_col,
                    'condition': ee.Filter.equals(**{
                        'leftField': 'system:index',
                        'rightField': 'system:index'
                    })
            }))
            s2_combined_col = s2_combined_col.map(lambda x: x.clip(target_bd))

            for cld_prb_id in range(0, num_cld_prb):
                if not s2_img_found_flag:
                    cld_prb_v = int(cld_prb_set[cld_prb_id])
                    s2_combined_col_tmp = s2_combined_col.map(lambda x: add_cloud_shadow_mask_infer(x, cld_prb_v)).map(computeQualityScore_infer)
                    s2_cloudless_col_BEST = s2_combined_col_tmp.filter(ee.Filter.lt('CLOUDY_PERCENTAGE', cloudFreeKeepThresh))

                    if s2_cloudless_col_BEST.size().gt(0):
                        s2_img_found_flag = True
                        s2_cloudless_col_BEST = s2_cloudless_col_BEST.sort("CLOUDY_PERCENTAGE", False)
                        s2_filtered = s2_combined_col_tmp.qualityMosaic('cloudShadowScore')
                        newC = ee.ImageCollection.fromImages([s2_filtered, s2_cloudless_col_BEST.mosaic()])
                        s2_img_cloudFree = ee.Image(newC.mosaic()).clip(target_bd)
                else:
                    break
        else:
            break
    
    s2_img_cloudFree = s2_img_cloudFree.select(["B4", "B3", "B2", "B8"])

    kernelSize = int(target_resolution / 10.0)
    overlapSize = overlapSize_ref[target_resolution]
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
    
    # ---------export the patched dataset into the format of TFRecord
    if destination == "Drive":
        records_task_config = {
            "collection": target_pt,
            "description": records_name,
            "folder": dst_dir,
            "fileFormat": "TFRecord",
        }
        records_task = ee.batch.Export.table.toDrive(**records_task_config)
        records_task.start()
    elif destination == "CloudStorage":
        records_task_config = {
            "collection": target_pt,
            "description": records_name,
            "bucket": BUCKET,
            "fileNamePrefix": DATA_FOLDER + "/" + records_name,
            "fileFormat": "TFRecord",
        }
        records_task = ee.batch.Export.table.toCloudStorage(**records_task_config)
        records_task.start()
    else:
        raise NotImplementedError("Unknown destination: {0}. Supported destination: ['Drive', 'CloudStorage']".format(destination))
    
    print("[Submit] TFRecords is to be exported at [{0}, {1}, {2}, {3}]".format(lon_min, lon_max, lat_min, lat_max))
    return records_task
    

def GBuildingMap_dataset(lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], year: int, GCS_config: Dict[str, str], target_resolution: int, dx=0.09, dy=0.09, precision=2, padding=0.02, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80, num_task_queue=10, num_queue_min=2, file_prefix=None):
    """Perform the 3D building information mapping using Google Cloud Service.

    Parameters
    ----------

    lon_min : Union[int, float]
        Minimum longitude of target mapping.
    lat_min : Union[int, float]
        Minimum latitude of target mapping.
    lon_max : Union[int, float]
        Maximum longitude of target mapping.
    lat_max : Union[int, float]
        Maximum latitude of target mapping.
    year : int
        Year of target mapping.
    GCS_config : Dict[str, str]
        Dictionary which specifies the configuration information of Google Cloud Service (GCS) for 3D building information mapping. An example is given as:
        GCS_config = {
            "SERVICE_ACCOUNT": "e-mail name of the GCS account",
            "GS_ACCOUNT_JSON": "path to the JSON key for the GCS service account",
            "BUCKET": "name of the Google Cloud Storage bucket",
            "DATA_FOLDER": "name of the folder under the Google Cloud Storage bucket prepared for storing intermediate datasets",
        }
    target_resolution : int
        Target resolution (in degrees) of the output mapping.
    dx : float
        Longitude coverage (in degrees) of sub-regions used for inference.
        The default is `0.09`.
    dy : float
        Latitude coverage (in degrees) of sub-regions used for inference.
        The default is `0.09`.
    precision : int
        The numerical precision of geographical coordinate calculation when dividing sub-regions.
        The default is `2`.
    padding : float
        Padding size outside the target region (in degrees) when exporting satellite images.
        The default is `0.02`.
    patch_size_ratio : int
        The ratio between the patch size of Sentinel-1/2 and SRTM data.
        The default is `1`.
    s2_cloud_prob_threshold : int
        The minimum threshold of Sentinel-2 cloud probability for filtering out cloudy pixels.
        The default is `20`.
    s2_cloud_prob_max : int
        The maximum threshold of Sentinel-2 cloud probability for filtering out cloudy pixels.
        The default is `80`.
    num_task_queue : int
        The number of concurrent Google Earth Engine's tasks for exporting datasets in the task queue.
        The default is `10`.
    num_queue_min : int
        The minimum number of tasks in the task queue. When the number of tasks in the task queue is less than `num_queue_min`, new tasks will be added to the task queue.
        The default is `2`.
    file_prefix : str
        The prefix of the temporary dataset name.
        The default is `None`.

    """
    BASE_INTERVAL = 1.0
    MIN_INTERVAL = 1.5
    GAMMA = 0.9

    SERVICE_ACCOUNT = GCS_config["SERVICE_ACCOUNT"]
    GS_ACCOUNT_JSON = GCS_config["GS_ACCOUNT_JSON"]

    # ---Google Cloud Storage bucket into which prediction datset will be written
    # BUCKET = GCS_config["BUCKET"]
    DATA_FOLDER = GCS_config["DATA_FOLDER"]

    # ------set a client for operations related to Google Cloud Storage
    # client_GCS = storage.Client(project=GCS_config["PROJECT_ID"])
    # client_GCS = storage.Client.from_service_account_json(GS_ACCOUNT_JSON)

    ee.Initialize(ee.ServiceAccountCredentials(SERVICE_ACCOUNT, GS_ACCOUNT_JSON))
    # ee.Initialize()

    # ------load Sentinel-1 and Sentinel-2 dataset
    DEM_dta = ee.Image("NASA/NASADEM_HGT/001").select(["elevation"]).float()
    # ------load Sentinel-1 dataset
    s1_ds = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
    s1_ds = s1_ds.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    s1_ds = s1_ds.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    s1_ds = s1_ds.filter(ee.Filter.eq('instrumentMode', 'IW'))
    s1_ds = s1_ds.select(["VV", "VH"])
    s1_year = min(2022, max(year, 2015))    # Sentinel-1 availability: from 2014-10 to 2022-12
    # ---------for northern hemisphere
    s1_time_start_N = datetime.datetime(year=s1_year, month=12, day=1).strftime("%Y-%m-%d")
    s1_time_end_N = datetime.datetime(year=s1_year+1, month=3, day=1).strftime("%Y-%m-%d")
    s1_ds_N = s1_ds.filter(ee.Filter.date(s1_time_start_N, s1_time_end_N))
    # ---------for southern hemisphere
    s1_time_start_S = datetime.datetime(year=s1_year, month=6, day=1).strftime("%Y-%m-%d")
    s1_time_end_S = datetime.datetime(year=s1_year, month=9, day=1).strftime("%Y-%m-%d")
    s1_ds_S = s1_ds.filter(ee.Filter.date(s1_time_start_S, s1_time_end_S))

    # ------load Sentinel-2 dataset
    s2_cld_threshold_base = 20
    s2_ds = ee.ImageCollection("COPERNICUS/S2_SR").filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', s2_cld_threshold_base))
    s2_year = min(2022, max(year, 2018))    # Sentinel-2 availability: from 2017-03 to 2022-12
    # ---------for northern hemisphere, we prefer: autumn > spring > summer
    # ---------for southern hemisphere, we prefer: spring > autumn > summer
    s2_time_start_aut = datetime.datetime(year=s2_year, month=9, day=1).strftime("%Y-%m-%d")
    s2_time_end_aut = datetime.datetime(year=s2_year, month=12, day=1).strftime("%Y-%m-%d")
    s2_ds_aut = s2_ds.filter(ee.Filter.date(s2_time_start_aut, s2_time_end_aut))

    s2_time_start_spr = datetime.datetime(year=s2_year, month=3, day=1).strftime("%Y-%m-%d")
    s2_time_end_spr = datetime.datetime(year=s2_year, month=6, day=1).strftime("%Y-%m-%d")
    s2_ds_spr = s2_ds.filter(ee.Filter.date(s2_time_start_spr, s2_time_end_spr))
    '''
    s2_time_start_sum = datetime.datetime(year=s2_year, month=6, day=1).strftime("%Y-%m-%d")
    s2_time_end_sum = datetime.datetime(year=s2_year, month=9, day=1).strftime("%Y-%m-%d")
    s2_ds_sum = s2_ds.filter(ee.Filter.date(s2_time_start_sum, s2_time_end_sum))

    s2_time_start_win = datetime.datetime(year=s2_year, month=12, day=1).strftime("%Y-%m-%d")
    s2_time_end_win = datetime.datetime(year=s2_year+1, month=3, day=1).strftime("%Y-%m-%d")
    s2_ds_win = s2_ds.filter(ee.Filter.date(s2_time_start_win, s2_time_end_win))    
    '''
    s2_cloudProb_ds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")

    # ------divide ROI into smaller patches
    num_lon = int(round((lon_max - lon_min) / dx, precision))
    if num_lon > 0:
        lon_min_sub = np.arange(lon_min, lon_min + num_lon * dx, dx)
    else:
        lon_min_sub = np.array([lon_min])
    xCell_num = len(lon_min_sub)

    num_lat = int(round((lat_max - lat_min) / dy, precision))
    if num_lat > 0:
        lat_max_sub = np.arange(lat_max, lat_max - num_lat * dy, -dy)
    else:
        lat_max_sub = np.array([lat_max])
    yCell_num = len(lat_max_sub)

    ee_taskList = []
    ee_coordsList = []
    # ee_suffixList = []
    for xId in range(0, xCell_num):
        for yId in range(0, yCell_num):
            lon_min_tmp = lon_min_sub[xId]
            lon_max_tmp = lon_min_tmp + dx
            lat_max_tmp = lat_max_sub[yId]
            lat_min_tmp = lat_max_tmp - dy

            # ------export patched datasets from Google Earth Engine (GEE) to Google Cloud Storage (GCS)
            if any([not isinstance(lon_min_tmp, int), not isinstance(lon_max_tmp, int), not isinstance(lat_min_tmp, int), not isinstance(lat_max_tmp, int)]):
                # on_min_f, lon_min_i = math.modf(lon_min_tmp)
                lon_min_str = str(round(lon_min_tmp, precision))
                # lat_min_f, lat_min_i = math.modf(lat_min_tmp)
                lat_min_str = str(round(lat_min_tmp, precision))
                # lon_max_f, lon_max_i = math.modf(lon_max_tmp)
                # lon_max_str = "%d.%d" % (lon_max_i, round(lon_max_f * 100))
                lon_max_str = str(round(lon_max_tmp, precision))
                # lat_max_f, lat_max_i = math.modf(lat_max_tmp)
                # lat_max_str = "%d.%d" % (lat_max_i, round(lat_max_f * 100))
                lat_max_str = str(round(lat_max_tmp, precision))
                file_suffix = "_".join([lon_min_str, lon_max_str, lat_min_str, lat_max_str])
            else:
                file_suffix = "_".join([str(lon_min_tmp), str(lon_max_tmp), str(lat_min_tmp), str(lat_max_tmp)])
            
            if 0.5 * (lat_max_tmp + lat_min_tmp) > 0:
                task_tmp = export_satData_GEE(DEM_dta, s1_ds_N, [s2_ds_aut, s2_ds_spr], s2_cloudProb_ds, GCS_config=GCS_config,
                                              lon_min=lon_min_tmp, lat_min=lat_min_tmp, lat_max=lat_max_tmp, lon_max=lon_max_tmp, year=year, target_resolution=target_resolution,
                                              dst_dir=DATA_FOLDER, precision=precision, destination="CloudStorage", file_prefix=file_prefix, padding=padding, patch_size_ratio=patch_size_ratio,
                                              s2_cloud_prob_threshold=s2_cloud_prob_threshold, s2_cloud_prob_max=s2_cloud_prob_max)
            else:
                task_tmp = export_satData_GEE(DEM_dta, s1_ds_S, [s2_ds_spr, s2_ds_aut], s2_cloudProb_ds, GCS_config=GCS_config,
                                              lon_min=lon_min_tmp, lat_min=lat_min_tmp, lat_max=lat_max_tmp, lon_max=lon_max_tmp, year=year, target_resolution=target_resolution,
                                              dst_dir=DATA_FOLDER, precision=precision, destination="CloudStorage", file_prefix=file_prefix, padding=padding, patch_size_ratio=patch_size_ratio,
                                              s2_cloud_prob_threshold=s2_cloud_prob_threshold, s2_cloud_prob_max=s2_cloud_prob_max)
            
            ee_taskList.append(task_tmp)
            ee_coordsList.append([lon_min_str, lon_max_str, lat_min_str, lat_max_str])

            if xId == xCell_num - 1 and yId == yCell_num - 1:
                final_flag = True
            else:
                final_flag = False

            if len(ee_taskList) >= num_task_queue or final_flag:
                task_infoList = [task.status() for task in ee_taskList]
                task_IDList = [tinfo["name"].split("/")[-1] for tinfo in task_infoList]
                task_stateList = [tinfo["state"] for tinfo in task_infoList]
                # task_msgList = ["" for task in task_infoList]
                task_runningID = [i for i in range(0, len(task_stateList)) if task_stateList[i] in ["UNSUBMITTED", "READY", "RUNNING", "CANCEL_REQUESTED"]]
                num_iter = 0
                while (len(task_runningID) > num_queue_min and not final_flag) or (len(task_runningID) > 0 and final_flag):
                    num_active = 0
                    task_finishedID = []
                    # ----only check the status of tasks which were previously active
                    print(task_runningID)
                    for loc in range(0, len(task_runningID)):
                        tid = task_runningID[loc]
                        tid_st = ee_taskList[loc].status()
                        # ------ref to: https://developers.google.com/earth-engine/guides/processing_environments
                        if tid_st["state"] not in ["UNSUBMITTED", "READY", "RUNNING", "CANCEL_REQUESTED"]:
                            task_finishedID.append(loc)
                            # task_stateList[tid] = tid_st["state"]
                            if tid_st["state"] != "COMPLETED":
                                # task_msgList[loc] = tid_st["error_message"]
                                print("*****[Error] TFRecords fails to be exported at [{0}] (TaskID = {1}) due to {2}*****".format(", ".join(ee_coordsList[loc]), task_IDList[tid], tid_st["error_message"]))
                            else:
                                print("*****[Success] TFRecords is exported at [{0}] (TaskID = {1})*****".format(", ".join(ee_coordsList[loc]), task_IDList[tid]))
                        else:
                            # print(tid_st["state"])
                            num_active += 1
                    
                    print(task_finishedID)
                    for loc in sorted(task_finishedID, reverse=True):
                        del task_runningID[loc]
                        del ee_taskList[loc]
                        del ee_coordsList[loc]
                    
                    if num_active <= 2:
                        t_sleep = MIN_INTERVAL
                    else:
                        t_sleep = max(BASE_INTERVAL * (GAMMA**num_iter) * num_active, MIN_INTERVAL)
                    print("[Iter. {0}] {1} tasks are still running. Sleep {2} s".format(num_iter, num_active, t_sleep))
                    num_iter += 1
                    time.sleep(t_sleep)
    

if __name__ == "__main__":
    # ---Google Cloud Service configuration
    parser = argparse.ArgumentParser(description="CNN Model Training")
    # dataset
    parser.add_argument("--service_account", type=str, help="the e-mail of Google Cloud Service account")
    parser.add_argument("--GS_json", type=str, help="json")
    # parser.add_argument("--project_id", type=str, help="the ID of the Google Cloud Project")
    parser.add_argument("--bucket_name", type=str, help="the name of the Google Cloud Storage bucket")
    parser.add_argument("--data_folder", type=str, help="the name of the folder in the Google Cloud Storage bucket")

    args = parser.parse_args()

    '''
    GCS_config = {
        "SERVICE_ACCOUNT": "273902675329-compute@developer.gserviceaccount.com",
        "BUCKET": "lyy-shafts",
        "DATA_FOLDER": "dataset_tmp",
    }
    '''
    GCS_config = {
        "SERVICE_ACCOUNT": args.service_account,
        "GS_ACCOUNT_JSON": args.GS_json,
        "BUCKET": args.bucket_name,
        "DATA_FOLDER": args.data_folder,
    }

    GBuildingMap_dataset(lon_min=-0.50, lat_min=51.00, lon_max=0.4, lat_max=51.90, year=2020, dx=0.09, dy=0.09, precision=3,
                            GCS_config=GCS_config,
                            target_resolution=100, num_task_queue=30, num_queue_min=2,
                            file_prefix=args.service_account[0:5], padding=0.01, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80)
