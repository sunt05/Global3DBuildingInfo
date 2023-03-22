from typing import Union, List, Dict
import datetime
import re
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


FEATURES = ["VV", "VH", "B4", "B3", "B2", "B8", "elevation"]


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
    print(len(dta))
    print(dta[0].shape)
    '''
    dta = list(imgDataset.as_numpy_iterator())
    print(dta[0][0][0, :, :, 0])
    print(dta[0][0].shape)
    plt.imshow(dta[0][0][0, :, :, 0], vmin=-11, vmax=-5, cmap="turbo")
    plt.show()
    '''
    print("Dataset parsed: {0}".format([" ".join(recordPathList)]))

    return imgDataset


createInferDataset(["testSample/___-0.5_-0.41_51.0_51.09_H100W100.tfrecord.gz"], target_resolution=100)
