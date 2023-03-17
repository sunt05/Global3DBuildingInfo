import os
import h5py
import argparse
import math
import numpy as np
import datetime
import pandas as pd
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt

import geopandas as gpd
from pyproj import Proj
from shapely.geometry import Polygon, shape


def getPopInfo_GPWv4(output_path, x_min, x_max, y_min, y_max, dx=0.09, dy=0.09, ratio=10):
  # ------ref to: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11
  # ------ref to: https://sedac.ciesin.columbia.edu/binaries/web/sedac/collections/gpw-v4/gpw-v4-documentation-rev11.pdf
  gpw_ds = gdal.Open("gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11_2020_30_sec_tif/gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev11_2020_30_sec.tif")
  gpw_dta = gpw_ds.ReadAsArray() * 1.0
  gpw_noData = gpw_ds.GetRasterBand(1).GetNoDataValue()
  gpw_dta = np.where(gpw_dta == gpw_noData, 0.0, gpw_dta)
  gpw_xMin, gpw_dx, _, gpw_yMax, _, gpw_dy = gpw_ds.GetGeoTransform()

  lon_min_list = []
  lon_max_list = []
  lat_min_list = []
  lat_max_list = []
  area_list = []
  num_pop_list = []
  pop_density_list = []

  x_csize = dx * ratio
  y_csize = dy * ratio

  #catch_flag = False
  xcMin_ex = np.linspace(x_min, x_max, num=int(round((x_max - x_min) / x_csize, 0)), endpoint=False)
  ycMin_ex = np.linspace(y_min, y_max, num=int(round((y_max - y_min) / y_csize, 0)), endpoint=False)

  for xc_min in xcMin_ex:
    xc_max = round(xc_min + x_csize, 7)     # ------round to 1 m
    #if not catch_flag:
    for yc_min in ycMin_ex:
      #if not catch_flag:
      yc_max = round(yc_min + y_csize, 7)   # ------round to 1 m      
      # ------generate central points of each potential ROI
      xloc = np.arange(xc_min, xc_max, dx)
      xCell_num = len(xloc)
      yloc = np.arange(yc_max, yc_min, -dy)
      yCell_num = len(yloc)
      
      lon_min_tmp = []
      lon_max_tmp = []
      lat_min_tmp = []
      lat_max_tmp = []
      area_tmp = []
      num_pop_tmp = []
      pop_density = []
      for xId in range(0, xCell_num):
        for yId in range(0, yCell_num):
          lon_min = xloc[xId]
          lon_max = xloc[xId] + dx
          lat_max = yloc[yId]
          lat_min = yloc[yId] - dy
          lon = [lon_min, lon_max, lon_max, lon_min]
          lat = [lat_max, lat_max, lat_min, lat_min]
          # ---------calculate the area of each potential ROI
          # ------------ref to: https://stackoverflow.com/questions/4681737/how-to-calculate-the-area-of-a-polygon-on-the-earths-surface-using-python
          pa = Proj("+proj=aea +lat_1={0} +lat_2={1} +lat_0={2} +lon_0={3}".format(lat_min, lat_max, 0.5 * (lat_min + lat_max), 0.5 * (lon_min + lon_max)))
          x_cr, y_cr = pa(lon, lat)
          cop = {"type": "Polygon", "coordinates": [zip(x_cr, y_cr)]}
          area = shape(cop).area / 1000.0 / 1000.0
          area_tmp.append(area)
          # ---------calculate the population of each potential ROI
          # ------------ref to: https://stackoverflow.com/questions/47404898/find-indices-of-raster-cells-that-intersect-with-a-polygon
          x_start = np.floor((lon_min - gpw_xMin) / gpw_dx)
          x_end = np.ceil((lon_max - gpw_xMin) / gpw_dx)
          y_start = np.floor((lat_max - gpw_yMax) / gpw_dy)
          y_end = np.ceil((lat_min - gpw_yMax) / gpw_dy)
          num_pop = np.sum(gpw_dta[int(y_start):int(y_end), int(x_start):int(x_end)])
          num_pop_tmp.append(num_pop)
          # ---------calculate the average population density
          pop_density.append(num_pop / area)
          print(num_pop / area)

          lon_min_tmp.append(lon_min)
          lon_max_tmp.append(lon_max)
          lat_min_tmp.append(lat_min)
          lat_max_tmp.append(lat_max)

      # ------store the information
      lon_min_list.append(lon_min_tmp)
      lon_max_list.append(lon_max_tmp)
      lat_min_list.append(lat_min_tmp)
      lat_max_list.append(lat_max_tmp)
      area_list.append(area_tmp)
      num_pop_list.append(num_pop_tmp)
      pop_density_list.append(pop_density)
  
  with h5py.File(output_path, "w") as hrsl_db:
    hrsl_db.create_dataset("lon_min", data=np.concatenate(lon_min_list, axis=0))
    hrsl_db.create_dataset("lon_max", data=np.concatenate(lon_max_list, axis=0))
    hrsl_db.create_dataset("lat_min", data=np.concatenate(lat_min_list, axis=0))
    hrsl_db.create_dataset("lat_max", data=np.concatenate(lat_max_list, axis=0))
    hrsl_db.create_dataset("area", data=np.concatenate(area_list, axis=0))
    hrsl_db.create_dataset("num_pop", data=np.concatenate(num_pop_list, axis=0))
    hrsl_db.create_dataset("pop_density", data=np.concatenate(pop_density_list, axis=0))

  
def globalPopExport():
  xMin_batch = np.array([0, 45, -45, 90, -90, 135, -135, -180], dtype=np.int32)
  xMax_batch = xMin_batch + 45
  yMin_batch = np.array([0, 27, 54, -27, -54, -81], dtype=np.int32)
  yMax_batch = np.array([27, 54, 81, 0, -27, -54], dtype=np.int32)

  for idx in range(0, len(xMin_batch)):
    xmin_b = xMin_batch[idx]
    xmax_b = xMax_batch[idx]
    for idy in range(0, len(yMin_batch)):
      ymin_b = yMin_batch[idy]
      ymax_b = yMax_batch[idy]
      output_path = "GPW_info/GPW_{0}_{1}_{2}_{3}.h5".format(xmin_b, xmax_b, ymin_b, ymax_b)
      # getPopInfo(output_path, xmin_b, xmax_b, ymin_b, ymax_b, dx=0.09, dy=0.09)
      getPopInfo_GPWv4(output_path, xmin_b, xmax_b, ymin_b, ymax_b, dx=0.09, dy=0.09)


def selectROI_pop(pop_info_path: str, pop_density_min=100):
  with h5py.File(pop_info_path, "r") as pop_db:
    lon_min_dta = pop_db["lon_min"][()]
    lon_max_dta = pop_db["lon_max"][()]
    lat_min_dta = pop_db["lat_min"][()]
    lat_max_dta = pop_db["lat_max"][()]

    pop_density_dta = pop_db["pop_density"][()]
    roi_loc = (pop_density_dta > pop_density_min)

    lon_min_roi = np.reshape(lon_min_dta[roi_loc], newshape=(-1, 1))
    lon_max_roi = np.reshape(lon_max_dta[roi_loc], newshape=(-1, 1))
    lat_min_roi = np.reshape(lat_min_dta[roi_loc], newshape=(-1, 1))
    lat_max_roi = np.reshape(lat_max_dta[roi_loc], newshape=(-1, 1))
    extent_roi = np.concatenate([lon_min_roi, lon_max_roi, lat_min_roi, lat_max_roi], axis=1)

    roi_geom = [Polygon([[ext[0], ext[3]], [ext[1], ext[3]], [ext[1], ext[2]], [ext[0], ext[2]], [ext[0], ext[3]]]) for ext in extent_roi]
    roi_gdf = gpd.GeoDataFrame({"geometry": roi_geom, "ROI_id": np.arange(0, len(roi_geom))})
    roi_gdf.to_file("testSample/ROI_GPWv4.shp")


def getPopStats(pop_info_dir: str, pD_min=50, pD_max=1000, pD_step=50):
  pop_h5_list = [f for f in os.listdir(pop_info_dir) if f.endswith(".h5")]

  pD_threshold = np.arange(pD_min, pD_max + pD_step, pD_step)
  pop_selectInfo = {"pop_density_min": [], "num_ROI": []}

  for pD in pD_threshold:
    num_roi = 0
    for f in pop_h5_list:
      with h5py.File(os.path.join(pop_info_dir, f), "r") as pop_db:
        pop_density_dta = pop_db["pop_density"][()]
        num_roi = num_roi + len(np.where(pop_density_dta > pD)[0])
      
    pop_selectInfo["pop_density_min"].append(pD)
    pop_selectInfo["num_ROI"].append(num_roi)
  
  pop_df = pd.DataFrame(pop_selectInfo)

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
  ax.plot(pop_selectInfo["pop_density_min"], pop_selectInfo["num_ROI"], zorder=1)
  ax.scatter(pop_selectInfo["pop_density_min"], pop_selectInfo["num_ROI"], c="red", marker="x", zorder=2)
  ax.set_xlabel("Minimum population density [persons / $\mathrm{km}^2$]")
  ax.set_ylabel("Number of selected ROIs")
  plt.tight_layout(pad=1.08)
  plt.savefig("HRSL_ROI.png", dpi=300)

  pop_df.to_csv("HRSL_ROI.csv", index=False) 


if __name__ == "__main__":
  '''
  sentinel2cloudFree_download(sample_csv="GEE_Download_2022_back.csv", dst_dir="Sentinel-2_export_CF", path_prefix="/Volumes/ForLyy/Temp/ReferenceData", 
                                padding=0.05, cloud_prob_threshold=30, dst="Drive")
  '''
  # getPopInfo(output_path="HRSL_info.h5", x_min=0, x_max=1.8, y_min=50, y_max=51.8, dx=0.09, dy=0.09)
  # getPopInfo_GPWv4(output_path="HRSL_info_GPWv4.h5", x_min=0, x_max=1.8, y_min=50, y_max=51.8, dx=0.09, dy=0.09)

  # globalPopExport()
  # getPopStats(pop_info_dir="GPWv4_info", pD_min=50, pD_max=1000, pD_step=50)
  getPopStats(pop_info_dir="HRSL_info", pD_min=50, pD_max=1000, pD_step=50)

  # selectROI_pop(pop_info_path="HRSL_info_GPWv4.h5", pop_density_min=300)