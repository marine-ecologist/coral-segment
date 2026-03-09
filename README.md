# Image Segmentation from Orthomosaics

Documented here is a three-stage workflow for automatically identifying and delineating individual discrete objects (here coral colonies) from .geotiff orthomosaics (here georeferenced drone imagery). 

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)


Each stage is a standalone Python script that can be run independently or chained together. The pipeline 1) splits the orthomosaic it into workable tiles, and 2) runs the Segment Anything Model (SAM) to detect individual coral colonies — outputting combined non-overlapping georeferenced polygons as GeoJSON.

For timeseries analysis of orthomosaics, the pipeline includes an interactive tool to georeference orthomosaics to an existing reference orthomosaic to align colonies in space and time prior to segmenting. 

Of the available tools, [TerraLab AI Segmentation](https://plugins.qgis.org/plugins/AI_Segmentation/) performs similarly in QGIS, and the `georeference` function in QGIS does similar to the interactive calibration yet is less efficient for processing largescale drone imagery.

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `gcp_calibrator.py` | Align a target raster to a reference orthomosaic using Ground Control Points |
| 2 | `split_geotiff.py` | Tile the aligned raster into manageable georeferenced chunks |
| 3 | `auto_coral_segment.py` | Detect coral colonies with SAM, resolve overlaps, and interactively refine |

---

## Requirements

```bash
pip install rasterio numpy Pillow scipy opencv-python matplotlib shapely torch
pip install git+https://github.com/facebookresearch/segment-anything.git
```

SAM requires a model checkpoint. Download one from the [SAM model zoo](https://github.com/facebookresearch/segment-anything#model-checkpoints) and place it in your working directory, home directory, or `~/Downloads/`:

| Model | Size | File |
|-------|------|------|
| ViT-H (recommended) | 2.4 GB | `sam_vit_h_4b8939.pth` |
| ViT-L | 1.2 GB | `sam_vit_l_0b3195.pth` |
| ViT-B | 375 MB | `sam_vit_b_01ec64.pth` |


---

## Stage 1 — GCP Calibrator

<img width="1200" alt="Screenshot 2026-03-09 at 11 31 29" src="https://github.com/user-attachments/assets/04a5df16-7306-4651-b41a-aa58bb438e3f" />

**`gcp_calibrator.py`** — Align a target raster to a reference orthomosaic by picking matching Ground Control Points in a synchronised split-screen viewer.

<img width="1200" alt="Screenshot 2026-03-09 at 11 31 21" src="https://github.com/user-attachments/assets/d8e9cdea-e7ea-45bf-a5ec-d8d803b190f7" />

gcp_calibrator is a useful "user-friendly" approach when the target raster (e.g. a quick-processed drone ortho) is spatially offset or distorted relative to a high-quality reference (e.g. a properly georeferenced orthomosaic or satellite basemap). gcp_calibrator allows manually selection of paired points in both images, and warps the target raster accordingly.

### Usage

```bash
python gcp_calibrator.py <reference.tif> <target.tif> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `<target>_corrected.tif` | Output path for the corrected raster |
| `-t`, `--transform` | `auto` | Geometric transform: `auto`, `affine`, `polynomial`, or `tps` |

### Controls

| Action | Input |
|--------|-------|
| Place GCP | Left-click (green on reference, red on target) |
| Zoom | Scroll / trackpad pinch (both panels) |
| Pan | Shift + drag, or right-click drag |
| Undo last point | Ctrl+Z |
| Clear all points | Ctrl+Shift+Z |
| Run calibration | Ctrl+R |
| Reset zoom | H |

### Transform methods

The `--transform` flag controls how the target is warped:

- **`auto`** — selects the best method based on GCP count (affine for 3, polynomial for 6+, TPS for 10+)
- **`affine`** — 6-parameter linear transform; needs ≥3 GCPs. Best for simple shift/rotation/scale corrections
- **`polynomial`** — 2nd-order polynomial; needs ≥6 GCPs. Handles moderate non-linear distortion
- **`tps`** — Thin Plate Spline; needs ≥10 GCPs. Handles complex local warping with exact interpolation through control points

### Example

```bash
# Align a low-quality drone ortho to a reference, using TPS with many GCPs
python gcp_calibrator.py reference_ortho.tif raw_drone.tif -o aligned_drone.tif -t tps
```

---

## Stage 2 — Split GeoTIFF

**`split_geotiff.py`** — divides a large orthomosaic into smaller georeferenced tiles. This keeps memory manageable for SAM (which must embed the entire image) and allows processing tiles independently or in parallel.

### Usage

```bash
python split_geotiff.py <input.tif> <output_folder/> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tile-size` | `50` | Tile dimension in metres (creates square tiles) |
| `--buffer` | `0` | Buffer overlap in metres around each tile |
| `--format` | `GTiff` | Output format (`GTiff` or `COG`) |
| `--compress` | `LZW` | Compression method (`LZW`, `DEFLATE`, `JPEG`, `PACKBITS`, `NONE`) |

### Choosing tile size

Tile size is a trade-off between SAM embedding time and colony context. Some guidelines:

- **10–25 m** — fast per tile, good for very dense reef. Risk of cutting colonies at edges
- **50 m** — reasonable default. ~2500 m² per tile
- **100+ m** — slower SAM embedding, but fewer edge effects

Using `--buffer` adds overlap between adjacent tiles, which helps avoid cutting colonies that straddle tile boundaries. A buffer of 2–5 m is usually sufficient. Overlapping detections can be merged later by spatial de-duplication in GIS.

### Outputs

Each tile is a fully georeferenced GeoTIFF with the original CRS and band structure preserved. A `tile_index.txt` CSV is written to the output folder with tile grid positions, pixel dimensions, metric dimensions, and geographic bounds.

### Example

```bash
# Split into 50 m tiles with 3 m overlap buffer
python split_geotiff.py aligned_drone.tif tiles/ --tile-size 50 --buffer 3

# Large tiles, COG format for cloud workflows
python split_geotiff.py aligned_drone.tif tiles/ --tile-size 100 --format COG
```

---

## Stage 3 — Automatic Coral Segmentation

**`auto_coral_segment.py`** — Runs SAM's automatic mask generator across an entire tile (or whole ortho), resolves all overlaps to produce a strictly planar partition (no polygon overlaps), then opens an interactive viewer for refinement.

**  — i) Automatic detection.** `SamAutomaticMaskGenerator` places a grid of point prompts across the image and generates candidate masks. Masks are filtered by area (min/max in cm², computed from the raster's ground resolution).

**  — ii) Overlap resolution.** Candidate masks are sorted by SAM's `predicted_iou` score (highest confidence first). A label map is built pixel-by-pixel: each pixel is assigned to the highest-scoring mask that claims it. Lower-ranked masks have their overlapping pixels removed. If a mask's remaining area falls below the minimum threshold after clipping, it is discarded. The result is a set of strictly non-overlapping masks — each pixel belongs to at most one colony.

**  — iii) Interactive refinement.** All auto-detected colonies are displayed as green overlays on the image. Allows:
- Click existing polygons to select and delete false positives
- Click empty areas to manually segment missed colonies (with automatic overlap prevention)
- Save progress to a project file and resume later

### Usage

```bash
python auto_coral_segment.py <input.tif> <output.geojson> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-area` | `17500` | Maximum colony area in cm² |
| `--min-area` | `5` | Minimum colony area in cm² (noise filter) |
| `--sam-checkpoint` | *(auto-detect)* | Path to SAM model weights |
| `--points-per-side` | `32` | Grid density for auto-detection (higher = more candidates, slower) |
| `--project` | — | Resume from a saved `.project.json` file |

### Controls (refinement phase)

| Action | Input |
|--------|-------|
| Select polygon | Left-click on an existing polygon (turns red) |
| Delete selected | `d`, Delete, or Backspace |
| Segment new colony | Left-click on empty area |
| Accept new mask | `a` |
| Reject new mask | `r` |
| Cycle mask options | `n` |
| Undo last action | `u` |
| Save progress | `s` |
| Quit and save | `q` |
| Adjust selectivity | `+` / `-` |
| Zoom / Pan | Matplotlib toolbar at bottom of window |

### Area thresholds

Area limits are specified in **cm²** and converted to pixels using the raster's actual ground resolution (read from the GeoTIFF transform). This works correctly for both projected CRS (metres) and geographic CRS (degrees, with approximate conversion).

The default maximum of 17,500 cm² (1.75 m²) is intended to reject background segments and substrate-scale masks that SAM sometimes produces. Adjust to suit your reef and imagery resolution.

### Overlap resolution detail

The label-map approach guarantees a planar partition:

```
For each candidate mask (sorted by confidence, descending):
    unclaimed_pixels = mask AND NOT already_claimed
    if unclaimed_pixels.area >= min_area:
        assign unclaimed_pixels to this colony
        mark them as claimed
    else:
        discard (too little remains after higher-ranked masks took priority)
```

When manually adding colonies during refinement, claimed pixels are automatically subtracted from SAM's output before display. You cannot create overlapping polygons.

### Outputs

- **GeoJSON** (`output.geojson`) — Feature collection with polygon geometries in the raster's CRS. Each feature includes `area_pixels`, `area_cm2`, `source` (`auto` or `manual`), and SAM `score`.
- **Project file** (`output.project.json`) — Full session state including RLE-encoded masks. Resume with `--project`.

### Examples

```bash
# Basic run with default settings
python auto_coral_segment.py tile_0001.tif tile_0001_corals.geojson

# Larger colonies permitted, denser point grid
python auto_coral_segment.py tile_0001.tif tile_0001_corals.geojson \
    --max-area 25000 --points-per-side 48

# Resume a previous session
python auto_coral_segment.py --project tile_0001_corals.project.json
```

---

## Full Workflow Example

```bash
# 1. Align target raster to reference
python gcp_calibrator.py reference_ortho.tif raw_drone.tif -o aligned.tif

# 2. Split into tiles
python split_geotiff.py aligned.tif tiles/ --tile-size 50 --buffer 3

# 3. Segment each tile
for tile in tiles/tile_*.tif; do
    base=$(basename "$tile" .tif)
    python auto_coral_segment.py "$tile" "results/${base}_corals.geojson"
done
```

After processing, merge all per-tile GeoJSON outputs in QGIS or with a script (e.g. `ogr2ogr -append`) and de-duplicate any colonies that appear in buffer overlap zones.

---

## Companion Tool: Interactive Coral Segmenter

**`interactive_coral_segment.py`** is a purely manual alternative to Stage 3 for cases where full control is needed — click individual colonies one at a time, accept or reject each, and export. It uses the same SAM point-prompt approach but without the automatic detection phase.

```bash
python interactive_coral_segment.py input.tif output.geojson
python interactive_coral_segment.py --project output.project.json  # resume
```

This is useful for small areas, ground-truthing, or validating auto-segmentation results.

---

## Notes

- **SAM model choice.** ViT-H gives the best segmentation quality but requires ~8 GB RAM. ViT-B is 6× smaller and still produces good results for well-separated colonies.
- **Tile size vs. detection quality.** Smaller tiles run faster but may clip large colonies at edges. Use `--buffer` to mitigate this.
- **Points per side.** The default grid of 32×32 = 1024 point prompts works well for most reef imagery. Increase to 48 or 64 for dense thickets where colonies are tightly packed, at the cost of longer auto-segmentation time.
- **Selectivity tuning.** During refinement, press `+`/`-` to raise or lower the IoU threshold for manual click-to-segment additions. Higher values produce tighter masks around individual colonies; lower values capture more of the colony margin.
- **Batch processing.** For large surveys, loop over tiles with a shell script (see example above) and skip the interactive refinement phase by saving auto-results directly (the GeoJSON is written even if you close the window immediately).
- **Coordinate systems.** All tools preserve the input raster's CRS. For best area accuracy, use a projected CRS (UTM) rather than geographic (EPSG:4326).
