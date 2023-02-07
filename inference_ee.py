from typing import Union, List, Dict
import datetime
import re
import subprocess
import time
import ee
import matplotlib.pyplot as plt
import tensorflow as tf
import rasterio
import rasterio.transform as rtransform

from utils import *
from DL_model import *


GS_UTILS_BIN = "/Users/lyy/Downloads/lib_added/google-cloud-sdk/bin/gsutil"
REGION = "us-central1"

# ---Cloud Storage bucket into which prediction datset will be written
BUCKET = "lyy_shafts"

DATA_FOLDER = "dataset_tmp"

STL_MODEL_FOLDER = "STL"
STL_MODEL_GCS_PREFIX = "gs://" + BUCKET + "/" + STL_MODEL_FOLDER

MTL_MODEL_FOLDER = "MTL"
MTL_MODEL_GCS_PREFIX = "gs://" + BUCKET + "/" + MTL_MODEL_FOLDER

# ---Directory specification
# MODEL_DIR = "gs://" + OUTPUT_BUCKET + "/{0}".format(MODEL)
# ------put the EEified model next to the trained model directory
#EEIFIED_DIR = "gs://" + OUTPUT_BUCKET + "/eeified_{0}".format(MODEL)


FEATURES = ["VV_p50", "VH_p50", "B4", "B3", "B2", "elevation"]


def export_satData_GEE(lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], year: int, target_resolution: int, dst_dir: str, destination="CloudStorage", file_prefix=None, padding=0.04, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80):
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
    ee.Initialize()

    if any([not isinstance(lon_min, int), not isinstance(lon_max, int), not isinstance(lat_min, int), not isinstance(lat_max, int)]):
        lon_min_f, lon_min_i = math.modf(lon_min)
        lon_min_str = "%d.%d" % (lon_min_i, round(lon_min_f * 100))
        lat_min_f, lat_min_i = math.modf(lat_min)
        lat_min_str = "%d.%d" % (lat_min_i, round(lat_min_f * 100))
        lon_max_f, lon_max_i = math.modf(lon_max)
        lon_max_str = "%d.%d" % (lon_max_i, round(lon_max_f * 100))
        lat_max_f, lat_max_i = math.modf(lat_max)
        lat_max_str = "%d.%d" % (lat_max_i, round(lat_max_f * 100))
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

                    bandnamelist = s2_img_cloudFree.bandNames().getInfo()
                    if len(bandnamelist) > 0:
                        s2_img_found_flag = True
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
    test_image = ee.Image.cat([s1_img, s2_img_cloudFree, DEM_dta]).float()
    img_task_config = {
        "image": test_image,
        "description": records_name,
        "folder": dst_dir,
        "scale": 10,
        "maxPixels": 1e13,
        "crs": "EPSG:4326"
    }

    img_task = ee.batch.Export.image.toDrive(**img_task_config)
    img_task.start()
    '''
    
    # ---------export the patched dataset into the format of TFRecord
    if destination == "Drive":
        records_task_config = {
            "collection": target_pt,
            "description": records_name,
            "folder": dst_dir,
            "fileFormat": "TFRecord",
        }
        records_task = ee.batch.Export.table.toDrive(**records_task_config)
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

    while records_task.active():
        time.sleep(10)

    if records_task.status()["state"] != "COMPLETED":
        print("[Error] TFRecords fails to be exported at [{0}, {1}, {2}, {3}]".format(lon_min, lon_max, lat_min, lat_max))
    else:
        print("[Success] TFRecords is exported at [{0}, {1}, {2}, {3}]".format(lon_min, lon_max, lat_min, lat_max))
    
    
def parseSatTFRecord(example_proto: tf.train.Example, target_resolution: int, patch_size_ratio=1):
    kernelSize = int(target_resolution / 10.0)
    overlapSize = overlapSize_ref[target_resolution]

    sentinel_psize = kernelSize + overlapSize
    dem_psize = int(sentinel_psize * patch_size_ratio)

    COLUMNS = [tf.io.FixedLenFeature(shape=[sentinel_psize, sentinel_psize], dtype=tf.float32) if k != "elevation" else tf.io.FixedLenFeature(shape=[dem_psize, dem_psize], dtype=tf.float32) for k in FEATURES]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))
    featParsed_ref = tf.io.parse_single_example(example_proto, FEATURES_DICT)

    return featParsed_ref


def toTupleImage(feature_dict: Dict[str, tf.Tensor]):
    featList = [feature_dict.get(k) for k in FEATURES]
    feat_stacked = tf.stack(featList, axis=0)
    # ------convert CHW to HWC
    feat_stacked = tf.transpose(feat_stacked, [1, 2, 0])
    # ------the y-axis of exported patches should be flipped
    feat_sentinel = tf.experimental.numpy.flip(feat_stacked[:, :, :-1], axis=0)
    feat_aux = tf.expand_dims(feat_stacked[:, :, -1], -1)
    feat = tf.concat([feat_sentinel, feat_aux], axis=-1)

    return feat


def createInferDataset(recordPathList: List[str], target_resolution: int, batch_size=1) -> tf.data.TFRecordDataset:
    imgDataset = tf.data.TFRecordDataset(recordPathList, compression_type="GZIP")
    imgDataset = imgDataset.map(lambda x: parseSatTFRecord(x, target_resolution), num_parallel_calls=1)

    imgDataset = imgDataset.map(toTupleImage).batch(batch_size)
    '''
    dta = list(imgDataset.as_numpy_iterator())
    print(dta[0][0][0, :, :, 0])
    print(dta[0][0].shape)
    plt.imshow(dta[0][0][0, :, :, 0], vmin=-11, vmax=-5, cmap="turbo")
    plt.show()
    '''

    return imgDataset
    

def GBuildingMap(lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], year: int, pretrained_model: Union[Dict[str, str], str], target_resolution: int, file_prefix=None, padding=0.02, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80, MTL=True, removed=True):
    # ------export patched datasets from Google Earth Engine (GEE) to Google Cloud Storage (GCS)
    export_satData_GEE(lon_min=lon_min, lat_min=lat_min, lat_max=lat_max, lon_max=lon_max, year=year, target_resolution=target_resolution,
                            dst_dir=DATA_FOLDER, destination="CloudStorage", file_prefix=file_prefix, padding=padding, patch_size_ratio=patch_size_ratio,
                            s2_cloud_prob_threshold=s2_cloud_prob_threshold, s2_cloud_prob_max=s2_cloud_prob_max)
    
    # ------prepare the TFRecords dataset on GCS
    ds_foloder = "'gs://{0}/{1}".format(BUCKET, DATA_FOLDER)
    record_bytes = subprocess.Popen("/bin/sh {0} ls {1}".format(GS_UTILS_BIN, ds_foloder), stdout=subprocess.PIPE).stdout.read()
    record_list = [f for f in record_bytes.decode().split("\n")[:-1] if f.endswith(".tfrecord.gz")]
    
    img_ds = createInferDataset(record_list, target_resolution)

    # ---------determine the information of dimensions of the dataset
    h, w = re.findall(pattern=r"H(\d+)W(\d+)", string=record_list[0])[0]
    h = int(h)
    w = int(w)
    
    # ------model configuration
    cuda_used = (cuda_used and torch.cuda.is_available())

    if MTL:
        # ------prepare the Tensorflow-based MTL models
        m = tf.keras.models.load_model(pretrained_model)
        '''
        m = model_SEResNetMTLAuxTF(input_channels=6, input_size=psize, aux_input_size=psize, in_plane=in_plane, num_block=num_block, num_aux=1,
                                        log_scale=False, cuda_used=cuda_used, trained_record=pretrained_model)
        '''
        footprint_dta, height_dta = m.predict(x=img_ds, verbose=1)
    else:
        # ------prepare the Tensorflow-based STL models
        m_footprint = tf.keras.models.load_model(pretrained_model["footprint"])
        '''
        m_footprint = model_SEResNetAuxTF(input_channels=6, input_size=psize, aux_input_size=psize, in_plane=in_plane, num_block=num_block, num_aux=1,
                                            log_scale=False, activation="sigmoid", cuda_used=cuda_used, trained_record=pretrained_model["footprint"])
        '''
        footprint_dta = m_footprint.predict(x=img_ds, verbose=1)

        res = np.reshape(res, newshape=(h, w))
        m_height = tf.keras.models.load_model(pretrained_model["height"])
        '''
        m_height = model_SEResNetAuxTF(input_channels=6, input_size=psize, aux_input_size=psize, in_plane=in_plane, num_block=num_block, num_aux=1,
                                            log_scale=False, activation="relu", cuda_used=cuda_used, trained_record=pretrained_model["height"])
        '''
        height_dta = m_height.predict(x=img_ds, verbose=1)
    
    footprint_dta = np.reshape(footprint_dta, newshape=(h, w))
    height_dta = np.reshape(height_dta, newshape=(h, w))

    # ------export predicted images to Google Cloud Storage
    if any([not isinstance(lon_min, int), not isinstance(lon_max, int), not isinstance(lat_min, int), not isinstance(lat_max, int)]):
        lon_min_f, lon_min_i = math.modf(lon_min)
        lon_min_str = "%d.%d" % (lon_min_i, round(lon_min_f * 100))
        lat_min_f, lat_min_i = math.modf(lat_min)
        lat_min_str = "%d.%d" % (lat_min_i, round(lat_min_f * 100))
        lon_max_f, lon_max_i = math.modf(lon_max)
        lon_max_str = "%d.%d" % (lon_max_i, round(lon_max_f * 100))
        lat_max_f, lat_max_i = math.modf(lat_max)
        lat_max_str = "%d.%d" % (lat_max_i, round(lat_max_f * 100))
        file_suffix = "_".join([lon_min_str, lon_max_str, lat_min_str, lat_max_str])
    else:
        file_suffix = "_".join([str(lon_min), str(lon_max_str), str(lat_min_str), str(lat_max_str)])

    output_geo_trans = rtransform.Affine(target_resolution, 0.0, lon_min, 0.0, -target_resolution, lat_max)

    footprint_name = "BF" + "_" + file_suffix
    with rasterio.open(footprint_name, "w+", width=w, height=h, count=1, crs="EPSG:4326", transform=output_geo_trans, dtype="float32") as out:
        out.write_band(1, footprint_dta)
    
    height_name = "BH" + "_" + file_suffix
    with rasterio.open(height_name, "w+", width=w, height=h, count=1, crs="EPSG:4326", transform=output_geo_trans, dtype="float32") as out:
        out.write_band(1, height_dta)
    
    # ------delete temporary TFRcords files on the fly
    if removed:
        subprocess.call("/bin/sh {0} rm {1}".format(GS_UTILS_BIN, " ".join(record_list)))
        

if __name__ == "__main__":
    # export_satData_GEE(lon_min=0, lat_min=51.2, lon_max=0.09, lat_max=51.29, year=2020, file_prefix="_", target_resolution=100, padding=0.01, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80)
    # createInferDataset(["testSample/__0.0_0.9_51.20_51.29_H100W100.tfrecord.gz"], target_resolution=100)
    pass