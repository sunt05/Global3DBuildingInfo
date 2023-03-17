var irSumThresh = 0.3;
var ndviWaterThresh = -0.1;
var erodePixels = 1.5;
var dilationPixels = 3;
var SR_BAND_SCALE = 1e4;
var cloudFreeKeepThresh = 1;
var BUCKET = "lyy-shafts";
var DATA_FOLDER = "dataset_tmp";


function arange(start, stop, step) {
  var values = [];
  for (var value = start; value < stop; value += step) {
    values.push(value);
  }
  return values;
}


/*function computeQualityScore(img) {
  // QualityScore is calculated by selecting the maximum value between the Cloud Score and Shadow Score for each pixel
  var score = img.select(['cloudScore']).max(img.select(['shadowScore']));

  score = score.reproject('EPSG:4326', null, 20).reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: ee.Kernel.square(5)
  });

  score = score.multiply(-1);

  return img.addBands(score.rename('cloudShadowScore'));
}

function dilatedErossion(score) {
  // Perform opening on the cloud scores
  score = score.reproject('EPSG:4326', null, 20)
    .focalMin({radius: erodePixels, kernelType: 'circle', iterations: 3})
    .focalMin({radius: dilationPixels, kernelType: 'circle', iterations: 3})
    .reproject('EPSG:4326', null, 20);

  return score;
}
    
function mergeCollection_LightGBM(imgC) {
  // Select the best images, which are below the cloud free threshold, sort them in reverse order (worst on top) for mosaicing
  var best = imgC.filter(ee.Filter.lt('CLOUDY_PERCENTAGE', cloudFreeKeepThresh)).sort('CLOUDY_PERCENTAGE_ROI', false);

  // Composites all the images in a collection, using a quality band as a per-pixel ordering function (use pixels with the HIGHEST score).
  var filtered = imgC.qualityMosaic('cloudShadowScore');

  // Add the quality mosaic to fill in any missing areas of the ROI which aren't covered by good images
  var newC = ee.ImageCollection.fromImages([filtered, best.mosaic()]);

  // Note that the `mosaic` method composites overlapping images according to their order in the collection (last, i.e., best, on top)
  return ee.Image(newC.mosaic());
}

function add_cloud_bands(img, cloud_prob_threshold) {
  if (cloud_prob_threshold === undefined) {
    cloud_prob_threshold = 50;
  }

  var cloud_prob = ee.Image(img.get("s2cloudless")).select("probability").divide(100).rename("cloudScore");
  // 1 for cloudy pixels and 0 for non-cloudy pixels
  var is_cloud = cloud_prob.gt(cloud_prob_threshold / 100.0).rename("cloudFlag");

  return img.addBands(ee.Image([cloud_prob, is_cloud]));
}*/


function addCloudBandsInfer(img, cloudProbThreshold) {
  cloudProbThreshold = cloudProbThreshold || 50;
  var cloudProb = ee.Image(img.get("s2cloudless")).select("probability").rename("cloudScore");
  // 1 for cloudy pixels and 0 for non-cloudy pixels
  var isCloud = cloudProb.gt(cloudProbThreshold).rename("cloudFlag");

  return img.addBands(ee.Image([isCloud]));
}

/*function add_shadow_bands(img) {
  var meanAzimuth = img.get('MEAN_SOLAR_AZIMUTH_ANGLE');
  var meanZenith = img.get('MEAN_SOLAR_ZENITH_ANGLE');

  var cloudMask = img.select(['cloudFlag']);

  // Find dark pixels
  var darkPixelsImg = img.select(['B8', 'B11', 'B12']).divide(SR_BAND_SCALE).reduce(ee.Reducer.sum());

  var ndvi = img.normalizedDifference(['B8', 'B4']);
  var waterMask = ndvi.lt(ndviWaterThresh);
  var darkPixels = darkPixelsImg.lt(irSumThresh);

  // Get the mask of pixels which might be shadows excluding water
  var darkPixelMask = darkPixels.and(waterMask.not());
  darkPixelMask = darkPixelMask.and(cloudMask.not());

  // Find where cloud shadows should be based on solar geometry
  // Convert to radians
  var azR = ee.Number(meanAzimuth).add(180).multiply(Math.PI).divide(180.0);
  var zenR = ee.Number(meanZenith).multiply(Math.PI).divide(180.0);

  // Find the shadows
  function func_uke(cloudHeight) {
    cloudHeight = ee.Number(cloudHeight);

    var shadowCastedDistance = zenR.tan().multiply(cloudHeight); // Distance shadow is cast
    var x = azR.sin().multiply(shadowCastedDistance).multiply(-1); // X distance of shadow
    var y = azR.cos().multiply(shadowCastedDistance).multiply(-1); // Y distance of shadow
    return img.select(['cloudScore']).displace(ee.Image.constant(x).addBands(ee.Image.constant(y)));
  }

  var cloudHeights = ee.List.sequence(200, 10000, 250);
  var shadows = cloudHeights.map(func_uke);
  var shadowMasks = ee.ImageCollection.fromImages(shadows);
  var shadowMask = shadowMasks.mean();

  // Create shadow mask
  shadowMask = dilatedErossion(shadowMask.multiply(darkPixelMask));

  var shadowScore = shadowMask.reduceNeighborhood({
    reducer: ee.Reducer.max(),
    kernel: ee.Kernel.square(1)
  });

  img = img.addBands(shadowScore.rename(['shadowScore']));

  return img;
}*/


function add_shadow_bands_infer(img, nir_dark_threshold, cloud_proj_distance) {
  nir_dark_threshold = nir_dark_threshold || 0.15;
  cloud_proj_distance = cloud_proj_distance || 1.5;

  // Identify the water pixels from the SCL band
  var not_water = img.select("SCL").neq(6);

  // Identify dark NIR pixels that are not water (potential cloud shadow pixels)
  var dark_pixels = img.select("B8").lt(nir_dark_threshold * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels');

  // Determine the direction to project cloud shadow from clouds
  var shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

  // Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
  // `directionalDistanceTransform` function is used to calculate the distance between 
  // the nearest cloudy pixels and current non-cloudy pixels along `shadow_azimuth` (<= CLD_PRJ_DIST*10)
  var cld_proj = img.select('cloudFlag').directionalDistanceTransform(shadow_azimuth, cloud_proj_distance * 10)
    .reproject({ 'crs': img.select(0).projection(), 'scale': 100 })
    .select('distance')
    .mask()
    .rename('cloud_transform');

  // Identify the intersection of dark pixels with cloud shadow projection.
  var shadows = cld_proj.multiply(dark_pixels).rename('shadowFlag');

  // Add dark pixels, cloud projection, and identified shadows as image bands.
  img = img.addBands(ee.Image([shadows]));

  return img;
}


function apply_cloud_shadow_mask_infer(img) {
  // Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
  var not_cld_shdw = img.select('cloudmask').not();

  // Subset reflectance bands and update their masks, return the result.
  return img.updateMask(not_cld_shdw);
}


/*function add_cloud_shadow_mask(img, cloud_prob_threshold) {
  var img_cloud = add_cloud_bands(img, cloud_prob_threshold);
  var img_cloud_shadow = add_shadow_bands(img_cloud);

  var imgPoly = ee.Algorithms.GeometryConstructors.Polygon(
    ee.Geometry(img.get("system:footprint")).coordinates()
  );
  var roi = ee.Geometry(img.get('ROI'));
  var intersection = roi.intersection(imgPoly, ee.ErrorMargin(0.5));
  var cloudMask = img_cloud_shadow.select("cloudFlag").clip(roi).rename("cloudMask");
  
  var cloudAreaImg = cloudMask.multiply(ee.Image.pixelArea());
  //var cloudAreaImg = cloudMask.multiply(100.0);
  var stats = cloudAreaImg.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: roi,
    scale: 10,
    maxPixels: 1e14
  });

  // area of cloudy pixels / area of Sentinel-2 images
  var cloudPercent = ee.Number(stats.get('cloudMask')).divide(imgPoly.area()).multiply(100);

  // area of intersection with the city boundary / whole area of the city boundary
  var coveragePercent = ee.Number(intersection.area()).divide(roi.area()).multiply(100);
  // area of cloudy pixels / area of the city boundary
  var cloudPercentROI = ee.Number(stats.get('cloudMask')).divide(roi.area()).multiply(100);

  img_cloud_shadow = img_cloud_shadow.set('CLOUDY_PERCENTAGE', cloudPercent);
  img_cloud_shadow = img_cloud_shadow.set('ROI_COVERAGE_PERCENT', coveragePercent);
  img_cloud_shadow = img_cloud_shadow.set('CLOUDY_PERCENTAGE_ROI', cloudPercentROI);
  
  return img_cloud_shadow;
}*/


function add_cloud_shadow_mask_infer(img, cloud_prob_threshold) {
  cloud_prob_threshold = cloud_prob_threshold || 50;
  var img_cloud = addCloudBandsInfer(img, cloud_prob_threshold);
  var img_cloud_shadow = add_shadow_bands_infer(img_cloud);

  var is_cld_shdw = img_cloud_shadow.select("cloudFlag").add(img_cloud_shadow.select("shadowFlag")).gt(0);
  is_cld_shdw = is_cld_shdw.focalMin(2).focalMax(5).reproject({crs: img.select([0]).projection(), scale: 20}).rename('cloudmask');

  img = img.addBands(is_cld_shdw);
  
  return img;
}


function apply_cld_shdw_mask_infer(img) {
  // Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
  var not_cld_shdw = img.select('cloudmask').Not();

  // Subset reflectance bands and update their masks, return the result.
  return img.select('B.*').updateMask(not_cld_shdw);
}


var lon_min = -0.50;
var lat_min = 51.00;
var lon_max = -0.45;
var lat_max = 51.05;
var year = 2020;
var dx = 0.09;
var dy = 0.09;
var padding = 0.02;
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

var xloc = [-0.49955,-0.49865,-0.49774999999999997,-0.49684999999999996,-0.49594999999999995,-0.49504999999999993,-0.4941499999999999,-0.4932499999999999,-0.4923499999999999,-0.4914499999999999,-0.4905499999999999,-0.48964999999999986,-0.48874999999999985,-0.48784999999999984,-0.4869499999999998,-0.4860499999999998,-0.4851499999999998,-0.4842499999999998,-0.4833499999999998,-0.48244999999999977,-0.48154999999999976,-0.48064999999999974,-0.47974999999999973,-0.4788499999999997,-0.4779499999999997,-0.4770499999999997,-0.4761499999999997,-0.4752499999999997,-0.47434999999999966,-0.47344999999999965,-0.47254999999999964,-0.4716499999999996,-0.4707499999999996,-0.4698499999999996,-0.4689499999999996,-0.4680499999999996,-0.46714999999999957,-0.46624999999999955,-0.46534999999999954,-0.46444999999999953,-0.4635499999999995,-0.4626499999999995,-0.4617499999999995,-0.4608499999999995,-0.45994999999999947,-0.45904999999999946,-0.45814999999999945,-0.45724999999999943,-0.4563499999999994,-0.4554499999999994,-0.4545499999999994,-0.4536499999999994,-0.4527499999999994,-0.45184999999999936,-0.45094999999999935,-0.45004999999999934];
var yloc = [51.049549999999996,51.048649999999995,51.04774999999999,51.04684999999999,51.04594999999999,51.04504999999999,51.04414999999999,51.043249999999986,51.042349999999985,51.04144999999998,51.04054999999998,51.03964999999998,51.03874999999998,51.03784999999998,51.036949999999976,51.036049999999975,51.03514999999997,51.03424999999997,51.03334999999997,51.03244999999997,51.03154999999997,51.030649999999966,51.029749999999964,51.02884999999996,51.02794999999996,51.02704999999996,51.02614999999996,51.02524999999996,51.024349999999956,51.023449999999954,51.02254999999995,51.02164999999995,51.02074999999995,51.01984999999995,51.01894999999995,51.018049999999945,51.017149999999944,51.01624999999994,51.01534999999994,51.01444999999994,51.01354999999994,51.01264999999994,51.011749999999935,51.010849999999934,51.00994999999993,51.00904999999993,51.00814999999993,51.00724999999993,51.00634999999993,51.005449999999925,51.004549999999924,51.00364999999992,51.00274999999992,51.00184999999992,51.00094999999992,51.000049999999916];
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


// set the name of the exported TFRecords file
var records_name = "_" + "H50W50";

var s1_ds = ee.ImageCollection("COPERNICUS/S1_GRD");
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
var s1_img = s1_ds.reduce(ee.Reducer.percentile([50])).clip(target_bd);

var s2_cld_threshold_base = 50;
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
  print(s2_img_found_flag);
  if (!s2_img_found_flag) {
    var period = period_list[i];
    var s2_time_start_str = s2_season[period][0];
    var s2_time_end_str = s2_season[period][1];
    var s2_time_start = ee.Date(s2_time_start_str);
    var s2_time_end = ee.Date(s2_time_end_str);

    var s2_sr_col = ee.ImageCollection("COPERNICUS/S2_SR")
      .filterBounds(target_bd)
      .filter(ee.Filter.date(s2_time_start, s2_time_end))
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', s2_cld_threshold_base))
      .map(function(image){return image.clip(target_bd)});

    var s2_cloudless_col = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
      .filterBounds(target_bd)
      .filter(ee.Filter.date(s2_time_start, s2_time_end))
      .map(function(image){return image.clip(target_bd)});

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
        var s2_combined_col_tmp = s2_combined_col.map(function(x) { return add_cloud_shadow_mask_infer(x, cld_prb_v); }).map(function(x) { return apply_cloud_shadow_mask_infer(x); });
        s2_img_cloudFree = s2_combined_col_tmp.median();
        // s2_img_cloudFree = s2_img_cloudFree.reproject('EPSG:4326', null, 10);
        s2_img_cloudFree = s2_img_cloudFree.clip(target_bd);
        
        var bandnamelist = s2_img_cloudFree.bandNames().getInfo();
        if (bandnamelist.length > 0) {
          s2_img_found_flag = true;
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

// Generate the neighborhood patches from Sentinel imagery
var sentinel_psize = kernelSize + overlapSize;
var sentinel_value_default = ee.List.repeat(-1e4, sentinel_psize);
sentinel_value_default = ee.List.repeat(sentinel_value_default, sentinel_psize);
var sentinel_kernel = ee.Kernel.fixed(sentinel_psize, sentinel_psize, sentinel_value_default);

var sentinel_patched = sentinel_image.neighborhoodToArray(sentinel_kernel);
var target_pt = sentinel_patched.sampleRegions({collection: target_pt, scale: 10, geometries: true});

print(target_pt);

var records_task_config = {
    collection: target_pt,
    description: records_name,
    bucket: BUCKET,
    fileNamePrefix: DATA_FOLDER + "/" + records_name,
    fileFormat: "TFRecord"
};

Export.table.toCloudStorage(records_task_config);