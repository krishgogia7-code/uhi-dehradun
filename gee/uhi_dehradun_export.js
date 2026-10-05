// ===========================================================================
// URBAN HEAT ISLAND ANALYSIS - DEHRADUN
// Google Earth Engine (Code Editor) export script
//
// WHAT THIS DOES (matches presentation slides 5-6):
//   1. Loads Landsat 8/9 Collection 2 Level 2 scenes over Dehradun
//      (pre-monsoon months March-June, multi-year, cloud-cover filtered)
//   2. Masks clouds / shadows / snow and applies the official scale factors
//   3. Computes LST (surface temperature, deg C), NDVI, NDBI, NDWI
//   4. Adds SRTM elevation + slope
//   5. EXPORT 1 (table): one row per pixel per date ->
//        date | satellite | lon | lat | ndvi | ndbi | ndwi | elevation | slope | lst_c
//      This CSV is what the Python steps (step1, step2, dashboard) use.
//   6. EXPORT 2 (image): median pre-monsoon composite map (GeoTIFF) for the
//      report / QGIS.
//
// HOW TO RUN (one-time setup):
//   1. Sign up at https://earthengine.google.com (free with a Google account)
//   2. Open https://code.earthengine.google.com
//   3. Paste this whole file into the editor and click RUN
//   4. Open the "Tasks" tab (top-right) and click RUN on both exports.
//   5. Files appear in Google Drive -> folder "GEE_UHI_Export"
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Study area: Dehradun city bounding box (degrees)
//    To use an exact city boundary instead, upload a shapefile as an asset
//    and replace `roi` with that table's geometry.
// ---------------------------------------------------------------------------
var roi = ee.Geometry.Rectangle([77.95, 30.24, 78.12, 30.38]);
Map.centerObject(roi, 11);
Map.addLayer(roi, {color: 'red'}, 'Dehradun ROI', false);

// ---------------------------------------------------------------------------
// 2. Landsat 8 + 9 Collection 2 Level 2 (USGS, surface reflectance + ST) [8]
// ---------------------------------------------------------------------------
var landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
    .filterBounds(roi)
    .filterDate('2019-01-01', '2026-07-01')            // multi-year
    .filter(ee.Filter.calendarRange(3, 6, 'month'))    // pre-monsoon: Mar-Jun
    .filter(ee.Filter.lt('CLOUD_COVER', 40));          // keep usable scenes

print('Scenes found:', landsat.size());

// ---------------------------------------------------------------------------
// 3. Cloud/shadow/snow mask + band scaling (official Collection 2 factors)
// ---------------------------------------------------------------------------
function maskAndScale(img) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 1).eq(0)   // dilated cloud
      .and(qa.bitwiseAnd(1 << 3).eq(0))     // cloud
      .and(qa.bitwiseAnd(1 << 4).eq(0))     // cloud shadow
      .and(qa.bitwiseAnd(1 << 5).eq(0));    // snow

  var sr = img.select('SR_B.').multiply(0.0000275).add(-0.2);           // reflectance 0-1
  var lstC = img.select('ST_B10').multiply(0.00341802).add(149.0)       // Kelvin
      .subtract(273.15);                                                // -> deg C

  return ee.Image(img
      .addBands(sr, null, true)
      .addBands(lstC.rename('lst_c'), null, true)
      .updateMask(clear)
      .copyProperties(img, ['system:time_start', 'SPACECRAFT_ID']));
}

// ---------------------------------------------------------------------------
// 4. Spectral indices (slides 2-5)
//    NDVI = (NIR-Red)/(NIR+Red)            vegetation  [2]
//    NDBI = (SWIR1-NIR)/(SWIR1+NIR)        built-up
//    NDWI = (Green-NIR)/(Green+NIR)        open water (excluded from ref.)
// ---------------------------------------------------------------------------
function addIndices(img) {
  var ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('ndvi');
  var ndbi = img.normalizedDifference(['SR_B6', 'SR_B5']).rename('ndbi');
  var ndwi = img.normalizedDifference(['SR_B3', 'SR_B5']).rename('ndwi');
  return img.addBands([ndvi, ndbi, ndwi]);
}

// SRTM elevation + derived slope (comparable-elevation rule, slide 5)
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation');
var slope = ee.Terrain.slope(dem).rename('slope');

var scenes = landsat.map(maskAndScale).map(addIndices);

// ---------------------------------------------------------------------------
// 5. EXPORT 1 - point samples table (the dataset for Python/ML)
//    Sampled every ~240 m to keep the CSV a manageable size.
// ---------------------------------------------------------------------------
function sampleScene(img) {
  var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
  var pts = img.select(['ndvi', 'ndbi', 'ndwi', 'lst_c'])
      .addBands(dem)
      .addBands(slope)
      .sample({
        region: roi,
        scale: 240,
        projection: 'EPSG:4326',
        geometries: true,
        tileScale: 8
      });
  return pts.map(function(p) {
    var xy = ee.List(ee.Geometry(p.geometry()).coordinates());
    return p.set({
      date: date,
      satellite: img.get('SPACECRAFT_ID'),
      lon: xy.get(0),
      lat: xy.get(1)
    });
  });
}

var samples = ee.FeatureCollection(scenes.map(sampleScene)).flatten();

Export.table.toDrive({
  collection: samples,
  description: 'UHI_Dehradun_Samples',
  folder: 'GEE_UHI_Export',
  fileNamePrefix: 'dehradun_lst_samples',
  fileFormat: 'CSV',
  selectors: ['date', 'satellite', 'lon', 'lat',
              'ndvi', 'ndbi', 'ndwi', 'elevation', 'slope', 'lst_c']
});

// ---------------------------------------------------------------------------
// 6. EXPORT 2 - median pre-monsoon composite map (for the report / QGIS)
// ---------------------------------------------------------------------------
var composite = scenes.median()
    .select(['lst_c', 'ndvi', 'ndbi', 'ndwi'])
    .addBands(dem)
    .addBands(slope);

Export.image.toDrive({
  image: composite,
  description: 'UHI_Dehradun_MedianComposite',
  folder: 'GEE_UHI_Export',
  region: roi,
  scale: 30,
  maxPixels: 1e9,
  fileNamePrefix: 'dehradun_median_premonsoon'
});

// Quick visual check layers
Map.addLayer(composite.select('lst_c'),
    {min: 20, max: 45, palette: ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']},
    'LST median (deg C)');
Map.addLayer(composite.select('ndvi'), {min: 0, max: 0.7, palette: ['brown', 'white', 'green']},
    'NDVI median', false);

print('Done. Open the Tasks tab (right) and run the 2 exports.');
