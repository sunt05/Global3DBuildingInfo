# Global3DBuildingInfo

A global dataset of 3D building information from Sentinel imagery

## Building Height and Footprint Prediction using Google Cloud Service

Here we implement a `GBuildingMap` function to streamline the process of building height and footprint prediction via Google Cloud Service (GCS) so that 3D building information can be directly retrieved from the execution of a single function without downloading Sentinel-1/2 images locally and then applying CNN models for inference.

However, it requires some additional configuration of GCS before inference. The following part will give a brief introduction of the `GBuildingMap` function's usage.

### Configuration for Google Cloud Service

The following steps are required for setting up a [Goolge Earth Engine enabled Cloud Project](https://developers.google.com/earth-engine/cloud/earthengine_cloud_project_setup):

1. Create a [Google Cloud Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) in the Google Cloud [console](https://console.cloud.google.com/cloud-resource-manager) for building height and footprint prediction.

2. Enable the [Earth Engine API](https://console.cloud.google.com/apis/library/earthengine.googleapis.com) for the project.

3. Set up a bucket in the [Google Cloud Storage (GCS)](https://cloud.google.com/storage) prepared for the storage of some intermediate exported datasets from Google Earth Engine. Please note that **the names of the created bucket, its folder for storing intermediate datasets** are required by the execution of `GBuildingMap` function. An example of the structure of the GCS's bucket can be given as follows:

```bash
    bucket-name/
    |-- dataset/
    |   |-- exported-dataset.tfrecord.gz
    |   |-- ...
```

4. Create a [Google Cloud Service's account](https://console.cloud.google.com/iam-admin/serviceaccounts/) for the project. If there is already an account, you can keep it without creating an additional one. Please note that **the e-mail name of the service account** is required by the execution of `GBuildingMap` function.

5. Create a private key in the format of JSON for the service account by clicking the menu for that account via **:** > **key** > **JSON**. Please download the JSON key file locally and the **path to the JSON key for the service account** is required by the execution of `GBuildingMap` function.

### Building Height and Footprint Prediction

```python {cmd}
from inference_ee import GBuildingMap

# ---Path to the folder which contains pretrained MTL models based on Tensorflow
pretrained_weight = "./DL_run/height/check_pt_senet_100m_MTL_TF_gpu"

# ---Google Cloud Service configuration
GCS_config = {
    "SERVICE_ACCOUNT": "e-mail name of the GCS account",
    "GS_ACCOUNT_JSON": "path to the JSON key for the GCS service account",
    "BUCKET": "name of the Google Cloud Storage bucket",
    "DATA_FOLDER": "name of the folder under the Google Cloud Storage bucket prepared for storing intermediate datasets",
}

GBuildingMap(lon_min=-0.50, lat_min=51.00, lon_max=0.4, lat_max=51.90, year=2020, dx=0.09, dy=0.09, precision=3, batch_size=512, 
                pretrained_model=pretrained_weight, GCS_config=GCS_config,
                target_resolution=100, num_task_queue=30, num_queue_min=2,
                file_prefix="_", padding=0.01, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80, MTL=True, 
                removed=True, output_folder="./results")
```

<!-- ## Integrating locally trained CNN with GEE

1. Filter Sentinel-1/2 images of 10 m resolution amd SRTM data of 30 m resolution on GEE.

2. Export satellite images and data to Google Cloud Storage (GCS) via `ee.batch.Export.image.toCloudStorage` in the format of [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord). See this [link](https://developers.google.com/earth-engine/guides/tfrecord#exporting-images) for detailed descriptions.

    For [ee.batch.Export.image.toCloudStorage](https://developers.google.com/earth-engine/apidocs/export-image-tocloudstorage), there are several critical arguments:

    - `scale` which specifies output resolution in meters per pixel.
    - `crs` which specifies CRS to use for the exported image.
    - `fileFormat` which should be `TFRecord`.
    - `formatOptions` which is a dictionary of string keys including:
        - `patchDimensions` which specifies x, y size of exported patches / dimensions tiled over the export area.
        - `kernelSize` which specifies buffer size of tiles resulting in overlap between neighboring patches.

    Since Sentinel-1/2 and SRTM are in different resolutions, we can have two options for exporting data:
    - Export Sentinel-1/2 and SRTM data into two separate TFRecord files and then combine them using some postprocessing scripts.
    - Resample SRTM to the same resolution with Sentinel-1/2 images, concatenate all data into a single image and then export it into a single TFRecord file. In this case, we need to parse and resample samples appropriately during the iteration of the dataset.

3. Perform predictions on the exported dataset using CNN models stored on GCS and loaded by `tf.keras.models.load_model`.
-->
