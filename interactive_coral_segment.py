#!/usr/bin/env python3
"""
interactive_coral_segment.py - Interactive coral colony segmentation with SAM

Click on coral colonies one by one, SAM segments them, accept/reject each,
and export as georeferenced GeoJSON/Shapefile.

Session state is auto-saved to a .project.json file alongside the output,
so you can resume later with --project.

Usage:
    python interactive_coral_segment.py input.tif output.geojson
    python interactive_coral_segment.py --project output.project.json    # resume

Controls:
    - Left Click: Mark a colony (SAM will segment it)
    - 'a' key: Accept current segmentation
    - 'r' key: Reject current segmentation
    - 'u' key: Undo last accepted polygon
    - 'c' key: Clear current segmentation
    - 's' key: Save current progress + project file
    - 'q' key: Quit and save
    - Zoom/Pan: Use toolbar buttons at bottom of window
    - '+'/'-': Adjust SAM sensitivity
"""

import sys
import json
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Install dependencies if needed
try:
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import rowcol, xy
except ImportError:
    print("Installing rasterio...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterio", "--break-system-packages"])
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import rowcol, xy

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "--break-system-packages"])
    import cv2

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.backend_bases import MouseButton
except ImportError:
    print("Installing matplotlib...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "--break-system-packages"])
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.backend_bases import MouseButton

try:
    from shapely.geometry import shape, Polygon, mapping
    from shapely.ops import unary_union
except ImportError:
    print("Installing shapely...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shapely", "--break-system-packages"])
    from shapely.geometry import shape, Polygon, mapping
    from shapely.ops import unary_union

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Installing segment-anything...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/segment-anything.git", "--break-system-packages"])
    import torch
    from segment_anything import sam_model_registry, SamPredictor


class InteractiveCoralSegmenter:
    """Interactive segmentation tool with SAM"""

    # ── Project file helpers (RLE for compact mask storage) ──

    @staticmethod
    def _mask_to_rle(mask):
        """Encode a boolean mask as run-length encoding for compact JSON storage."""
        pixels = mask.flatten()
        changes = np.diff(pixels.astype(np.int8))
        run_starts = np.where(changes != 0)[0] + 1
        run_starts = np.concatenate([[0], run_starts, [len(pixels)]])
        lengths = np.diff(run_starts).tolist()
        return {
            'counts': lengths,
            'start_value': int(pixels[0]),
            'shape': list(mask.shape)
        }

    @staticmethod
    def _rle_to_mask(rle):
        """Decode an RLE dict back to a boolean numpy mask."""
        h, w = rle['shape']
        vals = []
        current = rle['start_value']
        for length in rle['counts']:
            vals.extend([current] * length)
            current = 1 - current
        return np.array(vals, dtype=bool).reshape(h, w)

    def save_project(self, project_path=None):
        """Save full session state to a JSON project file for later resumption."""
        if project_path is None:
            project_path = self.output_path.with_suffix('.project.json')
        else:
            project_path = Path(project_path)

        polygons_data = []
        for mask, geo_polygon in self.accepted_polygons:
            polygons_data.append({
                'mask_rle': self._mask_to_rle(mask),
                'geometry': mapping(geo_polygon),
                'area_pixels': int(mask.sum()),
            })

        project = {
            'version': 2,
            'geotiff_path': str(self.geotiff_path),
            'output_path': str(self.output_path),
            'settings': {
                'pred_iou_thresh': self.pred_iou_thresh,
                'stability_score_thresh': self.stability_score_thresh,
                'max_area_ratio': self.max_area_ratio,
            },
            'accepted_polygons': polygons_data,
            'polygon_count': len(self.accepted_polygons),
        }

        with open(project_path, 'w') as f:
            json.dump(project, f, indent=2)

        print(f"  Project saved -> {project_path}")
        print(f"  ({len(self.accepted_polygons)} polygons, "
              f"resume with --project {project_path})")
        return project_path

    def load_project(self, project_path):
        """Restore session state from a project file."""
        project_path = Path(project_path)
        if not project_path.exists():
            print(f"  Project file not found: {project_path}")
            return False

        with open(project_path) as f:
            project = json.load(f)

        # Verify it matches the current raster
        saved_raster = Path(project.get('geotiff_path', ''))
        if saved_raster.name != self.geotiff_path.name:
            print(f"  WARNING: Project was for '{saved_raster.name}', "
                  f"current raster is '{self.geotiff_path.name}'")
            print(f"  Polygons may not align correctly!")

        # Restore settings
        settings = project.get('settings', {})
        self.pred_iou_thresh = settings.get('pred_iou_thresh', self.pred_iou_thresh)
        self.stability_score_thresh = settings.get('stability_score_thresh',
                                                     self.stability_score_thresh)
        self.max_area_ratio = settings.get('max_area_ratio', self.max_area_ratio)
        self.max_area_pixels = int(self.width * self.height * self.max_area_ratio)

        # Restore polygons
        count = 0
        for poly_data in project.get('accepted_polygons', []):
            try:
                mask = self._rle_to_mask(poly_data['mask_rle'])
                geo_polygon = shape(poly_data['geometry'])
                if not geo_polygon.is_valid:
                    geo_polygon = geo_polygon.buffer(0)

                self.accepted_polygons.append((mask, geo_polygon))

                # Recreate the display patch
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    epsilon = 0.005 * cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, epsilon, True)
                    coords = approx.reshape(-1, 2)
                    patch = MplPolygon(
                        coords, closed=True,
                        edgecolor='lime', facecolor='lime',
                        alpha=0.3, linewidth=2
                    )
                    self.ax.add_patch(patch)
                    self.polygon_patches.append(patch)

                count += 1
            except Exception as e:
                print(f"  Warning: failed to restore polygon {count}: {e}")

        self.update_status()
        self.fig.canvas.draw_idle()

        print(f"  Restored {count} polygons from project")
        print(f"  Settings: selectivity={self.pred_iou_thresh:.2f}, "
              f"max_area={self.max_area_ratio*100:.2f}%")
        return True
    
    def __init__(self, geotiff_path, output_path, sam_checkpoint=None,
                 max_area_ratio=0.01, project_path=None):
        self.geotiff_path = Path(geotiff_path)
        self.output_path = Path(output_path)
        self.max_area_ratio = max_area_ratio  # Max area as fraction of image
        
        # Load GeoTIFF
        print(f"Loading GeoTIFF: {geotiff_path}")
        self.src = rasterio.open(geotiff_path)
        self.image_data = self.src.read([1, 2, 3])
        self.image = np.transpose(self.image_data, (1, 2, 0))  # HWC format
        
        self.height, self.width = self.image.shape[:2]
        self.max_area_pixels = int(self.width * self.height * self.max_area_ratio)
        
        print(f"Image size: {self.width} x {self.height} pixels")
        print(f"CRS: {self.src.crs}")
        print(f"Max colony area: {self.max_area_pixels} pixels ({self.max_area_ratio*100:.1f}% of image)")
        
        # Normalize for display
        self.display_image = self.image.copy()
        for i in range(3):
            band = self.display_image[:, :, i]
            p2, p98 = np.percentile(band, (2, 98))
            self.display_image[:, :, i] = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
        self.display_image = self.display_image.astype(np.uint8)
        
        # Load SAM model
        print("\nLoading SAM model...")
        self.load_sam(sam_checkpoint)
        
        # State
        self.current_point = None
        self.current_masks = None  # Store all 3 masks
        self.current_mask_idx = 0   # Which mask to display
        self.current_polygon_patch = None
        self.click_marker = None
        self.accepted_polygons = []  # List of (mask, geo_polygon) tuples
        self.polygon_patches = []  # Matplotlib patches for display
        
        # SAM parameters
        self.pred_iou_thresh = 0.88  # Higher = more selective
        self.stability_score_thresh = 0.95  # Higher = more stable masks
        
        # Setup plot
        self.setup_plot()

        # Load project file if resuming
        self._project_path = project_path
        if project_path and Path(project_path).exists():
            print(f"\nResuming from project: {project_path}")
            self.load_project(project_path)
        
        print("\n" + "="*70)
        print("INTERACTIVE SEGMENTATION MODE")
        print("="*70)
        print("Controls:")
        print("  Left Click   → Select a coral colony (SAM will segment)")
        print("  'a' key      → Accept current segmentation")
        print("  'r' key      → Reject current segmentation")
        print("  'n' key      → Next mask option (cycles through 3 options)")
        print("  'u' key      → Undo last accepted polygon")
        print("  'c' key      → Clear current segmentation")
        print("  's' key      → Save progress + project file (resume later)")
        print("  'q' key      → Quit and save")
        print("  '+'/'-'      → Increase/decrease selectivity")
        print("\n  Zoom/Pan     → Use toolbar at bottom (zoom, pan, home buttons)")
        print("="*70)
        print(f"\nAccepted polygons: 0")
        print(f"SAM selectivity: {self.pred_iou_thresh:.2f}")
        print("Waiting for input...\n")
    
    def load_sam(self, checkpoint_path):
        """Load SAM model"""
        if checkpoint_path and Path(checkpoint_path).exists():
            model_type = "vit_h"  # Assume vit_h for custom checkpoint
            checkpoint = checkpoint_path
        else:
            # Try to find SAM checkpoint in common locations
            common_paths = [
                "sam_vit_h_4b8939.pth",
                "sam_vit_l_0b3195.pth",
                "sam_vit_b_01ec64.pth",
                Path.home() / "sam_vit_h_4b8939.pth",
                Path.home() / "Downloads/sam_vit_h_4b8939.pth",
                Path.home() / "Downloads/sam_vit_l_0b3195.pth",
            ]
            
            checkpoint = None
            for path in common_paths:
                if Path(path).exists():
                    checkpoint = str(path)
                    if "vit_h" in str(path):
                        model_type = "vit_h"
                    elif "vit_l" in str(path):
                        model_type = "vit_l"
                    else:
                        model_type = "vit_b"
                    break
            
            if not checkpoint:
                print("\n⚠ SAM checkpoint not found!")
                print("Please download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
                print("Recommended: sam_vit_h_4b8939.pth (ViT-H, 2.4GB)")
                print("\nPlace it in:")
                print("  - Current directory")
                print("  - Home directory")
                print("  - Downloads folder")
                print("\nOr specify with: --sam-checkpoint path/to/checkpoint.pth")
                sys.exit(1)
        
        print(f"Using SAM model: {model_type}")
        print(f"Checkpoint: {checkpoint}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        
        self.predictor = SamPredictor(sam)
        self.predictor.set_image(self.image)
        
        print("✓ SAM model loaded and image embedded")
    
    def setup_plot(self):
        """Setup matplotlib interactive plot with toolbar"""
        self.fig, self.ax = plt.subplots(figsize=(14, 11))
        self.ax.imshow(self.display_image)
        self.ax.set_title("Interactive Coral Segmentation - Use toolbar to zoom/pan", fontsize=12)
        self.ax.axis('on')  # Keep axis on for toolbar to work
        
        # Text for status (top-left)
        self.status_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=10,
            family='monospace'
        )
        
        # Instructions (top-right)
        instructions = (
            "a=accept | r=reject | n=next mask\n"
            "u=undo | s=save | q=quit\n"
            "+/- = selectivity"
        )
        self.ax.text(
            0.98, 0.98, instructions, transform=self.ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
            fontsize=9,
            family='monospace'
        )
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Enable toolbar
        plt.tight_layout()
    
    def on_click(self, event):
        """Handle mouse click"""
        if event.inaxes != self.ax:
            return
        
        # Only respond to left click when not in zoom/pan mode
        if event.button == MouseButton.LEFT:
            # Check if toolbar is in navigation mode
            toolbar = plt.get_current_fig_manager().toolbar
            if toolbar.mode != '':  # If in zoom or pan mode, ignore
                return
            
            x, y = int(event.xdata), int(event.ydata)
            
            # Check bounds
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return
            
            print(f"\n→ Clicked at pixel ({x}, {y})")
            print("  Running SAM segmentation...")
            
            # Clear previous segmentation
            self.clear_current_segmentation()
            
            # Store click point
            self.current_point = (x, y)
            
            # Run SAM with better parameters
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                multimask_output=True  # Get 3 mask options
            )
            
            # Filter out masks that are too large
            valid_masks = []
            valid_scores = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                area = mask.sum()
                if area <= self.max_area_pixels:
                    valid_masks.append(mask)
                    valid_scores.append(score)
                    print(f"  Mask {i+1}: {area:,} pixels (score: {score:.3f}) ✓")
                else:
                    print(f"  Mask {i+1}: {area:,} pixels (score: {score:.3f}) ✗ TOO LARGE")
            
            if len(valid_masks) == 0:
                print("  ⚠ All masks too large! Try clicking closer to colony center.")
                print("  Tip: Press '+' to increase selectivity, or '-' to decrease")
                self.current_point = None
                return
            
            self.current_masks = valid_masks
            self.current_mask_idx = 0
            
            print(f"  Using mask 1/{len(valid_masks)}")
            print("  Press 'n' to cycle through mask options")
            
            # Visualize
            self.show_current_mask()
            
            print("  Press 'a' to accept or 'r' to reject")
    
    def show_current_mask(self):
        """Display current segmentation mask"""
        if self.current_masks is None or len(self.current_masks) == 0:
            return
        
        mask = self.current_masks[self.current_mask_idx]
        
        # Convert mask to polygon for display
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            print("  ⚠ No contours found in mask")
            return
        
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 3:
            print("  ⚠ Contour too small")
            return
        
        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Clear old polygon
        if self.current_polygon_patch:
            self.current_polygon_patch.remove()
        
        # Create matplotlib polygon (yellow with transparency)
        polygon_coords = approx.reshape(-1, 2)
        self.current_polygon_patch = MplPolygon(
            polygon_coords,
            closed=True,
            edgecolor='yellow',
            facecolor='yellow',
            alpha=0.4,
            linewidth=2.5
        )
        self.ax.add_patch(self.current_polygon_patch)
        
        # Clear old marker
        if self.click_marker:
            self.click_marker.remove()
        
        # Show click point
        if self.current_point:
            self.click_marker = self.ax.plot(
                self.current_point[0], self.current_point[1], 
                'r+', markersize=20, markeredgewidth=3
            )[0]
        
        self.update_status()
        self.fig.canvas.draw_idle()
    
    def cycle_mask(self):
        """Cycle to next mask option"""
        if self.current_masks is None or len(self.current_masks) <= 1:
            print("  No other mask options")
            return
        
        self.current_mask_idx = (self.current_mask_idx + 1) % len(self.current_masks)
        mask = self.current_masks[self.current_mask_idx]
        
        print(f"\n→ Showing mask {self.current_mask_idx + 1}/{len(self.current_masks)}")
        print(f"  Area: {mask.sum():,} pixels")
        
        self.show_current_mask()
    
    def clear_current_segmentation(self):
        """Clear current segmentation display"""
        if self.current_polygon_patch:
            self.current_polygon_patch.remove()
            self.current_polygon_patch = None
        
        if self.click_marker:
            self.click_marker.remove()
            self.click_marker = None
        
        self.current_masks = None
        self.current_mask_idx = 0
        self.current_point = None
        self.fig.canvas.draw_idle()
    
    def accept_segmentation(self):
        """Accept current segmentation"""
        if self.current_masks is None:
            print("  No segmentation to accept")
            return
        
        mask = self.current_masks[self.current_mask_idx]
        
        print(f"\n✓ Accepting segmentation (mask {self.current_mask_idx + 1})")
        
        # Convert mask to georeferenced polygon
        geo_polygon = self.mask_to_geo_polygon(mask)
        
        if geo_polygon is None:
            print("  ✗ Failed to create polygon")
            return
        
        # Store
        self.accepted_polygons.append((mask.copy(), geo_polygon))
        
        # Change color to green (accepted)
        if self.current_polygon_patch:
            self.current_polygon_patch.set_edgecolor('lime')
            self.current_polygon_patch.set_facecolor('lime')
            self.current_polygon_patch.set_alpha(0.3)
            self.current_polygon_patch.set_linewidth(2)
            self.polygon_patches.append(self.current_polygon_patch)
            self.current_polygon_patch = None
        
        # Clear click marker
        if self.click_marker:
            self.click_marker.remove()
            self.click_marker = None
        
        # Clear current state
        self.current_masks = None
        self.current_mask_idx = 0
        self.current_point = None
        
        print(f"  Total accepted: {len(self.accepted_polygons)}")
        self.update_status()
        self.fig.canvas.draw_idle()
    
    def reject_segmentation(self):
        """Reject current segmentation"""
        if self.current_masks is None:
            print("  No segmentation to reject")
            return
        
        print("\n✗ Rejecting segmentation")
        self.clear_current_segmentation()
        self.update_status()
    
    def undo_last(self):
        """Undo last accepted polygon"""
        if len(self.accepted_polygons) == 0:
            print("  No polygons to undo")
            return
        
        print("\n↶ Undoing last polygon...")
        self.accepted_polygons.pop()
        
        if len(self.polygon_patches) > 0:
            patch = self.polygon_patches.pop()
            patch.remove()
        
        print(f"  Remaining: {len(self.accepted_polygons)}")
        self.update_status()
        self.fig.canvas.draw_idle()
    
    def adjust_selectivity(self, increase=True):
        """Adjust SAM selectivity"""
        if increase:
            self.pred_iou_thresh = min(0.99, self.pred_iou_thresh + 0.02)
            self.max_area_ratio = max(0.001, self.max_area_ratio * 0.8)
            print(f"\n↑ Increased selectivity: {self.pred_iou_thresh:.2f}")
        else:
            self.pred_iou_thresh = max(0.70, self.pred_iou_thresh - 0.02)
            self.max_area_ratio = min(0.1, self.max_area_ratio * 1.25)
            print(f"\n↓ Decreased selectivity: {self.pred_iou_thresh:.2f}")
        
        self.max_area_pixels = int(self.width * self.height * self.max_area_ratio)
        print(f"  Max area: {self.max_area_pixels:,} pixels ({self.max_area_ratio*100:.2f}% of image)")
        self.update_status()
    
    def mask_to_geo_polygon(self, mask):
        """Convert pixel mask to georeferenced polygon"""
        # Get contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 3:
            return None
        
        # Simplify
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to pixel coordinates
        pixel_coords = approx.reshape(-1, 2)
        
        # Transform to geographic coordinates
        geo_coords = []
        for x, y in pixel_coords:
            lon, lat = self.src.xy(y, x)  # Note: rasterio uses (row, col)
            geo_coords.append((lon, lat))
        
        # Close the polygon
        if geo_coords[0] != geo_coords[-1]:
            geo_coords.append(geo_coords[0])
        
        # Create Shapely polygon
        try:
            polygon = Polygon(geo_coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # Fix invalid polygons
            return polygon
        except Exception as e:
            print(f"  ⚠ Error creating polygon: {e}")
            return None
    
    def save_polygons(self, show_message=True):
        """Save accepted polygons to GeoJSON/Shapefile and project file"""
        if len(self.accepted_polygons) == 0:
            if show_message:
                print("\n⚠ No polygons to save!")
            return
        
        if show_message:
            print(f"\nSaving {len(self.accepted_polygons)} polygons...")
        
        # Create GeoJSON structure
        features = []
        for idx, (mask, geo_polygon) in enumerate(self.accepted_polygons):
            feature = {
                'type': 'Feature',
                'properties': {
                    'id': idx + 1,
                    'area_pixels': int(mask.sum()),
                    'area_sq_meters': geo_polygon.area if self.src.crs.is_projected else None
                },
                'geometry': mapping(geo_polygon)
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': str(self.src.crs)}
            },
            'features': features
        }
        
        # Save GeoJSON
        with open(self.output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        if show_message:
            print(f"✓ Saved to: {self.output_path}")
            print(f"  CRS: {self.src.crs}")
            print(f"  Features: {len(features)}")

        # Also save project file for resumption
        self.save_project()
    
    def update_status(self):
        """Update status text"""
        status_lines = [
            f"Accepted: {len(self.accepted_polygons)}",
            f"Selectivity: {self.pred_iou_thresh:.2f}",
            f"Max area: {self.max_area_ratio*100:.2f}%"
        ]
        
        if self.current_masks is not None:
            mask = self.current_masks[self.current_mask_idx]
            status_lines.append(f"Current: {mask.sum():,} px")
            status_lines.append(f"Option: {self.current_mask_idx+1}/{len(self.current_masks)}")
        
        self.status_text.set_text('\n'.join(status_lines))
    
    def on_key(self, event):
        """Handle keyboard input"""
        if event.key == 'a':
            self.accept_segmentation()
        elif event.key == 'r':
            self.reject_segmentation()
        elif event.key == 'n':
            self.cycle_mask()
        elif event.key == 'u':
            self.undo_last()
        elif event.key == 'c':
            self.clear_current_segmentation()
        elif event.key == 's':
            print("\n💾 Saving current progress...")
            self.save_polygons()
        elif event.key == '+' or event.key == '=':
            self.adjust_selectivity(increase=True)
        elif event.key == '-' or event.key == '_':
            self.adjust_selectivity(increase=False)
        elif event.key == 'q':
            print("\n" + "="*70)
            print("QUITTING")
            print("="*70)
            self.save_polygons()
            plt.close()
    
    def run(self):
        """Start interactive session"""
        plt.show()
        
        # If window closed without 'q', still save
        if len(self.accepted_polygons) > 0:
            print("\nWindow closed - saving polygons...")
            self.save_polygons()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Missing arguments")
        print("\nUsage:")
        print("  python interactive_coral_segment.py input.tif output.geojson")
        print("  python interactive_coral_segment.py input.tif output.geojson --project saved.project.json")
        print("\nOptional:")
        print("  --sam-checkpoint path/to/sam_model.pth")
        print("  --max-area-ratio 0.01  (max colony size as fraction of image, default 0.01)")
        print("  --project path/to/file.project.json  (resume a previous session)")
        sys.exit(1)
    
    # Support --project as first arg for quick resume
    project_path = None
    if '--project' in sys.argv:
        idx = sys.argv.index('--project')
        project_path = sys.argv[idx + 1]
        # If only --project given, load paths from the project file
        if len(sys.argv) == 3 or (len(sys.argv) > 3 and sys.argv[1] == '--project'):
            with open(project_path) as f:
                proj = json.load(f)
            input_path = proj['geotiff_path']
            output_path = proj['output_path']
            print(f"Resuming project: {project_path}")
            print(f"  Raster: {input_path}")
            print(f"  Output: {output_path}")
        else:
            input_path = sys.argv[1]
            output_path = sys.argv[2]
    else:
        if len(sys.argv) < 3:
            print("Error: need input.tif and output.geojson, or --project file.json")
            sys.exit(1)
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Parse optional arguments
    sam_checkpoint = None
    max_area_ratio = 0.01  # 1% of image by default
    
    if '--sam-checkpoint' in sys.argv:
        idx = sys.argv.index('--sam-checkpoint')
        sam_checkpoint = sys.argv[idx + 1]
    
    if '--max-area-ratio' in sys.argv:
        idx = sys.argv.index('--max-area-ratio')
        max_area_ratio = float(sys.argv[idx + 1])
    
    try:
        segmenter = InteractiveCoralSegmenter(
            input_path, 
            output_path, 
            sam_checkpoint,
            max_area_ratio=max_area_ratio,
            project_path=project_path
        )
        segmenter.run()
        
        print("\n✓ Session complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()