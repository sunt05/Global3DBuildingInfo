# Global3DBuildingInfo

A global dataset of 3D building information from Sentinel imagery

## Integrating locally trained CNN with GEE

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
