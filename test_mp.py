from typing import Union, List, Dict
import datetime
import re
import multiprocessing
import subprocess
import time
import ee
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import rasterio
import rasterio.transform as rtransform
from google.cloud import storage

from utils import *
from DL_model import *


REGION = "us-central1"

# ---Google Cloud Service settings
PROJECT_ID = "robotic-door-289313"
SERVICE_ACCOUNT = "273902675329-compute@developer.gserviceaccount.com"
# ------following keys can be generated at: https://console.cloud.google.com/iam-admin/serviceaccounts
#GS_OAUTH2_PRIVATE_KEY = "*******************************"
#GS_OAUTH2_CLIENT_EMAIL = "*******************************"

GS_ACCOUNT_JSON = "./gcKey/robotic-door-289313-a5a8eb3f5632.json"

# ---Google Cloud Storage bucket into which prediction datset will be written
BUCKET = "lyy-shafts"

DATA_FOLDER = "dataset_tmp"
OUTPUT_FOLDER = "results"
OUTPUT_local_FOLDER = "./results"
BH_OUTPUT_GCS_PREFIX = "gs://" + BUCKET + "/" + OUTPUT_FOLDER + "/" + "height"
BF_OUTPUT_GCS_PREFIX = "gs://" + BUCKET + "/" + OUTPUT_FOLDER + "/" + "footprint"

#STL_MODEL_FOLDER = "STL"
#STL_MODEL_GCS_PREFIX = "gs://" + BUCKET + "/" + STL_MODEL_FOLDER
STL_MODEL_local_PREFIX = "./DL_run"

#MTL_MODEL_FOLDER = "MTL"
#MTL_MODEL_GCS_PREFIX = "gs://" + BUCKET + "/" + MTL_MODEL_FOLDER
MTL_MODEL_local_PREFIX = "./DL_run"


# FEATURES = ["VV_p50", "VH_p50", "B4", "B3", "B2", "B8", "elevation"]
FEATURES = ["VV", "VH", "B4", "B3", "B2", "B8", "elevation"]


def export_satData_GEE(lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], year: int, target_resolution: int, dst_dir: str, precision=2, destination="CloudStorage", file_prefix=None, padding=0.02, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80, pid=None):
    S2_SCALE = 1.0

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
    xloc = np.arange(lon_min + spc * 0.5, lon_max, spc)
    yloc = np.arange(lat_max - spc * 0.5, lat_min, -spc)
    x, y = np.meshgrid(xloc, yloc)
    xy_coords = np.concatenate([x.flatten().reshape(-1, 1), y.flatten().reshape(-1, 1)], axis=1)
    target_pt = ee.FeatureCollection([ee.Geometry.Point(cr[0], cr[1]) for cr in xy_coords])

    # ------set the name of the exported TFRecords file
    if file_prefix is not None:
        records_name = file_prefix + "_" + file_suffix + "_" + "H{0}".format(len(yloc)) + "W{0}".format(len(xloc))
    else:
        records_name = "_" + file_suffix + "_" + "H{0}".format(len(yloc)) + "W{0}".format(len(xloc))

    # ------filter the NASADEM data
    DEM_dta = ee.Image("NASA/NASADEM_HGT/001").select(["elevation"]).float()
    DEM_dta = DEM_dta.clip(target_bd)
    
    # ------filter the Sentinel-1 data
    s1_ds = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
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
    # ------note that if we use difference aggregation ops, exported bands would have different suffix with "_p50"
    # s1_img = s1_ds.reduce(ee.Reducer.percentile([50]))
    # s1_img = s1_img.clip(target_bd)
    s1_img = s1_ds.mean().clip(target_bd)

    # ------transform VV/VH bands of Sentinel-1 images to corresponding backscattering coefficients for the compatibility with SHAFTS
    # s1_img = get_backscatterCoef_EE(s1_img)

    # ------filter the Sentinel-2 cloud-free data
    s2_cld_threshold_base = 50
    s2_cld_prob_step = 5
    cld_prb_set = np.arange(s2_cloud_prob_threshold, s2_cloud_prob_max + s2_cld_prob_step, s2_cld_prob_step)
    num_cld_prb = len(cld_prb_set)
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
            
            s2_combined_col = s2_combined_col.map(lambda x: x.clip(target_bd))

            for cld_prb_id in range(0, num_cld_prb):
                if not s2_img_found_flag:
                    cld_prb_v = int(cld_prb_set[cld_prb_id])
                    s2_combined_col_tmp = s2_combined_col.map(lambda x: add_cloud_shadow_mask_infer(x, cld_prb_v)).map(apply_cloud_shadow_mask_infer)
                    
                    if s2_combined_col_tmp.size().gt(0):
                        s2_img_found_flag = True
                        s2_img_cloudFree = s2_combined_col_tmp.mean()
                        s2_img_cloudFree = s2_img_cloudFree.clip(target_bd)
                else:
                    break
        else:
            break
    
    s2_img_cloudFree = s2_img_cloudFree.select(["B4", "B3", "B2", "B8"])
    # ------rescale R/G/B/NIR bands of Sentinel-2 to [0, 255] for the compatibility with SHAFTS
    #s2_img_cloudFree = get_normalizedImage_EE(s2_img_cloudFree)

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
    
    # ---------export center points for sampling patches
    
    # ---------export the patched dataset into the format of TFRecord
    if destination == "Drive":
        records_task_config = {
            "collection": target_pt,
            "description": records_name,
            "folder": dst_dir,
            "fileFormat": "TFRecord",
        }
        records_task = ee.batch.Export.table.toDrive(**records_task_config)
        # records_task.start()
    elif destination == "CloudStorage":
        records_task_config = {
            "collection": target_pt,
            "description": records_name,
            "bucket": BUCKET,
            "fileNamePrefix": DATA_FOLDER + "/" + records_name,
            "fileFormat": "TFRecord",
        }
        records_task = ee.batch.Export.table.toCloudStorage(**records_task_config)
        # records_task.start()
    else:
        raise NotImplementedError("Unknown destination: {0}. Supported destination: ['Drive', 'CloudStorage']".format(destination))
    
    print("[Submit] TFRecords is to be exported at [{0}, {1}, {2}, {3}]".format(lon_min, lon_max, lat_min, lat_max))
    
    # return records_task
    if pid is None:
        return records_task
    else:
        return {"pid": pid, "task": records_task, "coords": [lon_min, lon_max, lat_min, lat_max]}
    
    
def parseSatTFRecord(example_proto: tf.train.Example, target_resolution: int, patch_size_ratio=1):
    kernelSize = int(target_resolution / 10.0)
    overlapSize = overlapSize_ref[target_resolution]

    sentinel_psize = kernelSize + overlapSize
    dem_psize = int(sentinel_psize * patch_size_ratio)

    COLUMNS = [tf.io.FixedLenFeature(shape=[sentinel_psize, sentinel_psize], dtype=tf.float32) if k != "elevation" else tf.io.FixedLenFeature(shape=[dem_psize, dem_psize], dtype=tf.float32) for k in FEATURES]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))
    featParsed_ref = tf.io.parse_single_example(example_proto, FEATURES_DICT)

    return featParsed_ref


def rgb_rescale_tf(dta: tf.Tensor, q1=0.98, q2=0.02, vmin=0.0, vmax=1.0):
    vmin = vmin * 1.0
    vmax = vmax * 1.0
    val_min = tf.experimental.numpy.min(dta, axis=(0, 1))
    val_min = tf.expand_dims(tf.expand_dims(val_min, axis=0), axis=0)

    val_high = tfp.stats.percentile(dta, q1 * 100, axis=[0, 1])
    # val_high = tf.expand_dims(tf.expand_dims(val_high, axis=0), axis=0)
    val_high = tf.reshape(val_high, shape=[1, 1, -1])
    val_low = tfp.stats.percentile(dta, q2 * 100, axis=[0, 1])
    # val_low = tf.expand_dims(tf.expand_dims(val_low, axis=0), axis=0)
    val_low = tf.reshape(val_low, shape=[1, 1, -1])
    
    dta_rescale = (dta - val_min) * (vmax - vmin) / (val_high - val_low) + vmin

    dta_clipped = tf.clip_by_value(dta_rescale, vmin, vmax)
    # ------set NaN value to be vmin
    dta_clipped = tf.where(~tf.math.is_finite(dta_clipped), vmin, dta_clipped)

    return dta_clipped


def get_backscatterCoef_tf(raw_s1_dta: tf.Tensor):
    # coef = tf.math.pow(10.0, raw_s1_dta / 10.0)
    # coef = tf.where(coef > 1.0, 1.0, coef)
    coef = tf.clip_by_value(raw_s1_dta, 0.0, 1.0)
    return coef


def toTupleImage(feature_dict: Dict[str, tf.Tensor]):
    featList = [feature_dict.get(k) for k in FEATURES]
    feat_stacked = tf.stack(featList, axis=0)
    # ------convert CHW to HWC
    feat_stacked = tf.transpose(feat_stacked, [1, 2, 0])

    # ------the y-axis of exported patches should be flipped
    feat_sentinel = tf.experimental.numpy.flip(feat_stacked[:, :, :-1], axis=0)
    # feat_sentinel = feat_stacked[:, :, :-1]
    # ---------Note that for SHAFTS v202203, the required data type of input Sentinel's bands are UINT8 -> tf.float32.
    feat_s1 = tf.cast(tf.cast(get_backscatterCoef_tf(feat_stacked[:, :, 0:2]) * 255, tf.uint8), tf.float32) / 255.0
    feat_s2 = tf.cast(tf.cast(rgb_rescale_tf(feat_sentinel[:, :, 2:], vmin=0, vmax=255), tf.uint8), tf.float32) / 255.0

    feat_aux = tf.expand_dims(feat_stacked[:, :, -1], -1)

    feat = tf.concat([feat_s1, feat_s2, feat_aux], axis=-1)

    return feat


def createInferDataset(recordPathList: List[str], target_resolution: int, batch_size=1) -> tf.data.TFRecordDataset:
    imgDataset = tf.data.TFRecordDataset(recordPathList, compression_type="GZIP")
    imgDataset = imgDataset.map(lambda x: parseSatTFRecord(x, target_resolution), num_parallel_calls=1)

    imgDataset = imgDataset.map(toTupleImage).batch(batch_size)
    dta = list(imgDataset.as_numpy_iterator())
    print(dta[0][0].shape)
    '''
    dta = list(imgDataset.as_numpy_iterator())
    print(dta[0][0][0, :, :, 0])
    print(dta[0][0].shape)
    plt.imshow(dta[0][0][0, :, :, 0], vmin=-11, vmax=-5, cmap="turbo")
    plt.show()
    '''
    print("Dataset parsed: {0}".format([" ".join(recordPathList)]))

    return imgDataset
    

def GBuildingMap(lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], year: int, pretrained_model: Union[Dict[str, str], str], target_resolution: int, dx=0.09, dy=0.09, precision=2, batch_size=16, file_prefix=None, padding=0.02, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80, MTL=True, removed=True, num_parallel=1, num_task_queue=10, num_cpu=2):
    num_cpu = min(num_cpu, multiprocessing.cpu_count())

    # ------divide ROI into smaller patches
    num_lon = int(np.floor((lon_max - lon_min) / dx))
    if num_lon > 0:
        lon_min_sub = np.arange(lon_min, lon_min + num_lon * dx, dx)
    else:
        lon_min_sub = np.array([lon_min])
    xCell_num = len(lon_min_sub)

    num_lat = int(np.floor((lat_max - lat_min) / dy))
    if num_lat > 0:
        lat_max_sub = np.arange(lat_max, lat_max - num_lat * dy, -dy)
    else:
        lat_max_sub = np.array([lat_max])
    yCell_num = len(lat_max_sub)

    ee_taskList = []
    ee_coordsList = []
    ee_argsList = []
    pid_c = 0
    pool = multiprocessing.Pool(processes=num_cpu)
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
            
            ee_args_tmp = (lon_min_tmp, lon_max_tmp, lat_min_tmp, lat_max_tmp, year, target_resolution, DATA_FOLDER, precision, "CloudStorage", file_prefix, padding, patch_size_ratio, s2_cloud_prob_threshold, s2_cloud_prob_max, pid_c)
            ee_argsList.append(ee_args_tmp)
            pid_c += 1
            
            while len(ee_argsList) >= num_cpu:
                task_res = pool.starmap(export_satData_GEE, ee_argsList)
                pool.close()
                pool.join()

                task_res = sorted(task_res, key=lambda x: x["pid"])
                for task_tmp in task_res:
                    task_tmp["task"].start()
                    time.sleep(0.1)
                    ee_taskList.append(task_tmp["task"])
                    ee_coordsList.append(task_tmp["coords"])
                ee_argsList.clear()
                pid_c = 0
                pool = multiprocessing.Pool(processes=num_cpu)

    

if __name__ == "__main__":
    # export_satData_GEE(lon_min=0, lat_min=51.2, lon_max=0.09, lat_max=51.29, year=2020, file_prefix="_", target_resolution=100, padding=0.01, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80)
    # createInferDataset(["testSample/__0.0_0.9_51.20_51.29_H100W100.tfrecord.gz"], target_resolution=100)
    # pretrained_weight = {"height": "check_pt_senet_100m_TF_gpu", "footprint": "check_pt_senet_100m_TF_gpu"}
    # pretrained_weight = {"height": "check_pt_senet_100m_MTL_TF_gpu", "footprint": "check_pt_senet_100m_MTL_TF_gpu"}
    pretrained_weight = "height/check_pt_senet_100m_MTL_TF_gpu"

    # x_min=0, x_max=1.8, y_min=50, y_max=51.8
    GBuildingMap(lon_min=-0.50, lat_min=51.00, lon_max=-0.05, lat_max=51.45, year=2020, dx=0.09, dy=0.09, precision=3, batch_size=512, pretrained_model=pretrained_weight, target_resolution=100, num_task_queue=12,
                    file_prefix="_", padding=0.02, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80, MTL=True, removed=True, num_cpu=4)