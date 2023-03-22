var cloudFreeKeepThresh = 2;
var cloudScale = 30;

var irSumThresh = 0.3;
var ndviWaterThresh = -0.1;
var erodePixels = 1.5;
var dilationPixels = 3;
var SR_BAND_SCALE = 1e4;
var cloudFreeKeepThresh = 1;
var BUCKET = "lyy-shafts";
var DATA_FOLDER = "dataset_tmp";

var lon_min = -0.455;
var lat_min = 51.045;
var lon_max = -0.41;
var lat_max = 51.09;
var year = 2020;
var dx = 0.09;
var dy = 0.09;
var padding = 0.015;
var target_resolution = 100;
var patch_size_ratio = 1;
var s2_cloud_prob_threshold = 20;
var s2_cloud_prob_max = 80;

var point_left_top = [lon_min, lat_max];
var point_right_top = [lon_max, lat_max];
var point_right_bottom = [lon_max, lat_min];
var point_left_bottom = [lon_min, lat_min];

dx = padding;
dy = padding;
point_left_top = [point_left_top[0] - dx, point_left_top[1] + dy];
point_right_top = [point_right_top[0] + dx, point_right_top[1] + dy];
point_right_bottom = [point_right_bottom[0] + dx, point_right_bottom[1] - dy];
point_left_bottom = [point_left_bottom[0] - dx, point_left_bottom[1] - dy];
var target_bd = ee.Geometry.Polygon([point_left_top, point_right_top, point_right_bottom, point_left_bottom, point_left_top]);

// set the sample point
var spc = 0.0009;

var xloc = [-0.45455, -0.45365, -0.45275, -0.45185, -0.45095, -0.45005, -0.44915, -0.44825, -0.44735, -0.44645, -0.44555, -0.44465, -0.44375, -0.44285, -0.44195, -0.44105, -0.44015, -0.43925, -0.43835, -0.43745, -0.43655, -0.43565, -0.43475, -0.43385, -0.43295, -0.43205, -0.43115, -0.43025, -0.42935, -0.42845, -0.42755, -0.42665, -0.42575, -0.42485, -0.42395, -0.42305, -0.42215, -0.42125, -0.42035, -0.41945, -0.41855, -0.41765, -0.41675, -0.41585, -0.41495, -0.41405, -0.41315, -0.41225, -0.41135, -0.41045];
var yloc = [51.08955, 51.08865, 51.08775, 51.08685, 51.08595, 51.08505, 51.08415, 51.08325, 51.08235, 51.08145, 51.08055, 51.07965, 51.07875, 51.07785, 51.07695, 51.07605, 51.07515, 51.07425, 51.07335, 51.07245, 51.07155, 51.07065, 51.06975, 51.06885, 51.06795, 51.06705, 51.06615, 51.06525, 51.06435, 51.06345, 51.06255, 51.06165, 51.06075, 51.05985, 51.05895, 51.05805, 51.05715, 51.05625, 51.05535, 51.05445, 51.05355, 51.05265, 51.05175, 51.05085, 51.04995, 51.04905, 51.04815, 51.04725, 51.04635, 51.04545];
var xy_coords = [];

var xy_coords = [];

xloc.forEach(function(x) {
  yloc.forEach(function(y) {
    xy_coords.push([x, y]);
  });
});

var target_pt = ee.FeatureCollection(xy_coords.map(function (coord) {
  var point = ee.Feature(ee.Geometry.Point(coord), {})
  return point
}));


function arange(start, stop, step) {
  var values = [];
  for (var value = start; value < stop; value += step) {
    values.push(value);
  }
  return values;
}


function addCloudBandsInfer(img, cloudProbThreshold) {
  cloudProbThreshold = cloudProbThreshold || 50;
  var cloudProb = ee.Image(img.get("s2cloudless")).select("probability").rename("cloudScore");
  var isCloud = cloudProb.gt(cloudProbThreshold).rename("cloudFlag");

  return img.addBands(ee.Image([cloudProb, isCloud]));
}


function add_shadow_bands_infer(img, nir_dark_threshold, cloud_proj_distance) {
  nir_dark_threshold = nir_dark_threshold || 0.15;
  cloud_proj_distance = cloud_proj_distance || 1.5;

  var not_water = img.select("SCL").neq(6);

  var dark_pixels = img.select("B8").lt(nir_dark_threshold * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels');

  var shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

  var cld_proj = img.select('cloudFlag').directionalDistanceTransform(shadow_azimuth, cloud_proj_distance * 10)
    .reproject({ 'crs': img.select(0).projection(), 'scale': 100 })
    .select('distance')
    .mask()
    .rename('cloud_transform');

  var shadows = cld_proj.multiply(dark_pixels).rename('shadowFlag');

  img = img.addBands(ee.Image([shadows]));

  return img;
}


function apply_cloud_shadow_mask_infer(img) {
  var not_cld_shdw = img.select('cloudmask').not();

  return img.updateMask(not_cld_shdw);
}


function computeQualityScore_infer(img) {
  var score = img.select(['cloudScore']).max(img.select(['shadowFlag']));

  score = score.reproject('EPSG:4326', null, 30).reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: ee.Kernel.square(1)
  });

  score = score.multiply(-1);

  return img.addBands(score.rename('cloudShadowScore'));
}


function add_cloud_shadow_mask_infer(img, cloud_prob_threshold) {
  cloud_prob_threshold = cloud_prob_threshold || 50;
  var img_cloud = addCloudBandsInfer(img, cloud_prob_threshold);
  var img_cloud_shadow = add_shadow_bands_infer(img_cloud);

  var imgPoly = ee.Algorithms.GeometryConstructors.Polygon(
    ee.Geometry(img.get("system:footprint")).coordinates()
  );

  var cloudMask = img_cloud_shadow.select("cloudFlag").add(img_cloud_shadow.select("shadowFlag")).gt(0);
  cloudMask = (cloudMask.focalMin(1.5).focalMax(1.5).reproject({ crs: img.select([0]).projection(), scale: cloudScale }).rename('cloudmask'));
  var cloudAreaImg = cloudMask.multiply(ee.Image.pixelArea());

  var stats = cloudAreaImg.reduceRegion({ reducer: ee.Reducer.sum(), scale: cloudScale, maxPixels: 1e14 });

  var cloudPercent = ee.Number(stats.get('cloudmask')).divide(imgPoly.area()).multiply(100);
  img_cloud_shadow = img_cloud_shadow.set('CLOUDY_PERCENTAGE', cloudPercent);

  return img_cloud_shadow;
}


function apply_cld_shdw_mask_infer(img) {
  var not_cld_shdw = img.select('cloudmask').Not();

  return img.select('B.*').updateMask(not_cld_shdw);
}


var records_name = "_" + "H50W50";

var s1_ds = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT");
s1_ds = s1_ds.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'));
s1_ds = s1_ds.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
s1_ds = s1_ds.filter(ee.Filter.eq('instrumentMode', 'IW'));
s1_ds = s1_ds.select(["VV", "VH"]);

var s1_year = Math.min(2022, Math.max(year, 2015)); // Sentinel-1 availability: from 2014-10 to 2022-12
var s1_time_start, s1_time_end;
if (0.5 * (lat_max + lat_min) > 0) {
  s1_time_start = s1_year.toString() + "-12-01";
  s1_time_end = (s1_year + 1).toString() + "-03-01";
} else {
  s1_time_start = s1_year.toString() + "-06-01";
  s1_time_end = s1_year.toString() + "-09-01";
}
s1_ds = s1_ds.filter(ee.Filter.date(s1_time_start, s1_time_end)).filterBounds(target_bd);
var s1_img = s1_ds.mean().clip(target_bd);
// var s1_img = s1_ds.product().log10().multiply(10).divide(s1_ds.count()).clip(target_bd);


var s2_cld_threshold_base = 30;
var s2_cld_prob_step = 5;
var s2_year = Math.min(2022, Math.max(year, 2018)); // Sentinel-2 availability: from 2017-03 to 2022-12

var s2_season = {
  "spring": [s2_year.toString() + "-03-01", s2_year.toString() + "-06-01"],
  "summer": [s2_year.toString() + "-06-01", s2_year.toString() + "-09-01"],
  "autumn": [s2_year.toString() + "-09-01", s2_year.toString() + "-12-01"],
  "winter": [s2_year.toString() + "-12-01", (s2_year + 1).toString() + "-03-01"],
};

var period_list = ["autumn", "spring", "summer", "winter"];

var s2_img_found_flag = false;
var s2_img_cloudFree = null;

var cld_prb_set = arange(s2_cloud_prob_threshold, s2_cloud_prob_max, s2_cld_prob_step);
var num_cld_prb = cld_prb_set.length;

for (var i = 0; i < period_list.length; i++) {
  if (!s2_img_found_flag) {
    var period = period_list[i];
    var s2_time_start_str = s2_season[period][0];
    var s2_time_end_str = s2_season[period][1];
    var s2_time_start = ee.Date(s2_time_start_str);
    var s2_time_end = ee.Date(s2_time_end_str);

    var s2_sr_col = ee.ImageCollection("COPERNICUS/S2_SR")
      .filterBounds(target_bd)
      .filter(ee.Filter.date(s2_time_start, s2_time_end))
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', s2_cld_threshold_base));

    var s2_cloudless_col = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
      .filterBounds(target_bd)
      .filter(ee.Filter.date(s2_time_start, s2_time_end));

    var s2_combined_col = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply({
      primary: s2_sr_col,
      secondary: s2_cloudless_col,
      condition: ee.Filter.equals({
        leftField: 'system:index',
        rightField: 'system:index'
      })
    }));

    s2_combined_col = s2_combined_col.map(function(x) {return x.clip(target_bd);});
    for (var cld_prb_id = 0; cld_prb_id < num_cld_prb; cld_prb_id++) {
      if (!s2_img_found_flag) {
        var cld_prb_v = cld_prb_set[cld_prb_id];
        var s2_combined_col_tmp = s2_combined_col.map(function(x) { return add_cloud_shadow_mask_infer(x, cld_prb_v); }).map(computeQualityScore_infer);
        
        var s2_cloudless_col_BEST = s2_combined_col_tmp.filter(ee.Filter.lt('CLOUDY_PERCENTAGE', cloudFreeKeepThresh));
        if (s2_cloudless_col_BEST.size().gt(0)){
          s2_img_found_flag = true;
          s2_cloudless_col_BEST = s2_cloudless_col_BEST.sort("CLOUDY_PERCENTAGE", false);
          var s2_filtered = s2_combined_col_tmp.qualityMosaic('cloudShadowScore');
          var newC = ee.ImageCollection.fromImages([s2_filtered, s2_cloudless_col_BEST.mosaic()]);
          var s2_img_cloudFree = ee.Image(newC.mosaic()).clip(target_bd);
        }
      } else {
        break;
      }
    }
  } else {
    break;
  }
}

s2_img_cloudFree = s2_img_cloudFree.select(["B4", "B3", "B2", "B8"]);

var kernelSize = Math.round(target_resolution / 10.0);
var overlapSize = 10;
var sentinel_image = ee.Image.cat([s1_img, s2_img_cloudFree]).float();

var sentinel_psize = kernelSize + overlapSize;
var sentinel_value_default = ee.List.repeat(-1e4, sentinel_psize);
sentinel_value_default = ee.List.repeat(sentinel_value_default, sentinel_psize);
var sentinel_kernel = ee.Kernel.fixed(sentinel_psize, sentinel_psize, sentinel_value_default);

var sentinel_patched = sentinel_image.neighborhoodToArray(sentinel_kernel);
// var sentinel_patched = s1_img.neighborhoodToArray(sentinel_kernel);
var target_pt = sentinel_patched.sampleRegions({collection: target_pt, scale: 10, geometries: true});

print(target_pt);