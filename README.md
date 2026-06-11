# ChangeMaster Ultimate — Phase 1 (Foundation)

**English** | [العربية](#العربية)

Offline satellite-image change detection for Windows (runs on Linux too for CI).
Phase 1 delivers the foundation: hardware adaptation, persistent configuration,
UTF-8 rotating logs, a unified multi-format I/O engine, tiled access for giant
rasters, and a sensor-profile system with automatic satellite detection.

## Features

- **Hardware detection & tiers** — CPU/RAM/disk/GPU probing classifies the
  machine (`low` / `medium` / `high` / `workstation`) and recommends worker
  counts and tile sizes. Works with or without `psutil`/`nvidia-smi`.
- **Persistent config** — JSON in `%APPDATA%\ChangeMaster` (Windows) or
  `~/.config/ChangeMaster` (Linux), atomic saves, recent-files list.
- **Rotating logs** — 5 MB × 3 files, UTF-8 (Arabic-safe), plus console output.
- **Bilingual errors** — every exception carries English + Arabic messages.
- **Unified I/O engine** with graceful degradation (missing optional
  dependencies never crash the app — formats are simply marked unavailable):

  | Format | Reader | Requires |
  |---|---|---|
  | PNG / JPEG / BMP | `simple_reader` | Pillow (always available) |
  | GeoTIFF / BigTIFF / JPEG2000 / ENVI | `raster_reader` | rasterio |
  | HDF5 | `hdf_reader` | h5py |
  | NetCDF | `netcdf_reader` | netCDF4 |
  | Sentinel `.SAFE` products | `safe_reader` | rasterio |
  | Landsat (folder or `.tar`) | `landsat_reader` | rasterio |

- **Tiled access** — process 100,000×100,000-pixel images in memory-bounded
  tiles with optional overlap.
- **Writers** — CRS/transform-preserving GeoTIFF output and always-available
  PNG quick-look export with percentile contrast stretch.
- **Sensor profiles** — 12 JSON profiles (Sentinel-1/2, Landsat 5/7/8/9,
  MODIS, WorldView, Pléiades, SPOT, PlanetScope, Generic) with band
  wavelengths/aliases and filename-based auto-detection.

## Installation

```bash
pip install -e .            # minimal (Pillow + NumPy)
pip install -e ".[full]"    # all formats (rasterio, h5py, netCDF4, psutil)
pip install -e ".[dev]"     # test tooling
```

## CLI tools

```bash
python scripts/titan_info.py            # hardware report + formats table
python scripts/titan_info.py --json
python scripts/titan_inspect.py <path>  # unified metadata + detected sensor
python scripts/titan_inspect.py <path> --json
```

## Python API

```python
from changemaster import open_image, TiledReader, write_geotiff, SensorRegistry

with open_image("scene.tif") as reader:
    meta = reader.metadata
    profile = SensorRegistry.detect_from_metadata(meta)
    for tile, pixels in TiledReader(reader, tile_size=1024):
        ...  # process each tile
```

## Running tests

```bash
python -m pytest tests/ --cov=changemaster
```

All test data is generated programmatically — no binary assets in the repo.

---

## العربية

برنامج **ChangeMaster Ultimate** لكشف التغيرات بين صور الأقمار الصناعية،
يعمل أوفلاين 100% على ويندوز (ويعمل على لينكس لأغراض الاختبار الآلي).
المرحلة الأولى تقدم الأساس الكامل للبرنامج.

### المزايا

- **كشف العتاد وتصنيفه** — فحص المعالج والذاكرة والقرص وكرت الشاشة وتصنيف
  الجهاز (منخفض / متوسط / عالي / محطة عمل) مع توصيات لعدد العمليات وحجم البلاطات.
- **إعدادات دائمة** — ملف JSON في `%APPDATA%\ChangeMaster` على ويندوز.
- **سجلات دوّارة** — ملفات 5 ميجابايت × 3 بترميز UTF-8 تدعم العربية بالكامل.
- **أخطاء ثنائية اللغة** — كل خطأ يحمل رسالة بالإنجليزية والعربية.
- **محرك قراءة موحد** — يدعم PNG/JPEG/BMP (دائماً)، وGeoTIFF/JPEG2000/ENVI
  وHDF5 وNetCDF ومنتجات Sentinel SAFE وLandsat (مجلد أو tar) عند توفر
  الاعتماديات الاختيارية — وإن لم تتوفر، يستمر البرنامج بالميزات المتاحة
  ولا ينهار أبداً.
- **قراءة مجزأة (Tiles)** — معالجة صور عملاقة بحجم 100,000×100,000 بكسل
  ضمن حدود الذاكرة المتاحة.
- **كتابة المخرجات** — GeoTIFF محافظ على نظام الإحداثيات والتحويل الجغرافي،
  وتصدير PNG سريع يعمل دائماً.
- **بروفايلات المستشعرات** — 12 بروفايل مع كشف تلقائي للقمر من اسم الملف.

### أدوات سطر الأوامر

```bash
python scripts/titan_info.py             # تقرير العتاد وجدول الصيغ
python scripts/titan_inspect.py <مسار>   # فحص صورة وعرض بياناتها الوصفية
```

### تشغيل الاختبارات

```bash
python -m pytest tests/ --cov=changemaster
```

كل بيانات الاختبار تُولّد برمجياً — لا ملفات ضخمة في المستودع.
