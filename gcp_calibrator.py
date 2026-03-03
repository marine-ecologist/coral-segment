#!/usr/bin/env python3
"""
GCP-Based Raster Calibration Tool  (tkinter, synchronised dual-view)
=====================================================================
Split-screen viewer for selecting Ground Control Points (GCPs) between a
high-quality reference raster (left) and a target raster (right), then
applying geometric correction to align the target to the reference.

Both panels are **geographically synchronised**: zoom and pan affect both
views simultaneously so the same map location stays centred in each panel.

Usage:
    python gcp_calibrator.py <reference_raster> <target_raster> [--output <path>]

Controls:
    Left-click             : place GCP (ref panel → green, tgt panel → red)
    Scroll / trackpad pinch: zoom in/out (both panels, centred on cursor)
    Shift + drag           : pan (both panels, no point placed)
    Ctrl+Z                 : undo last point
    Ctrl+Shift+Z           : clear all points
    Ctrl+R                 : run calibration
    H                      : reset zoom to fit both rasters

Requirements:
    pip install rasterio numpy Pillow scipy

Packaging:
    pip install pyinstaller
    pyinstaller --onefile --windowed gcp_calibrator.py
"""

import sys
import argparse
import platform
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import rasterio
from rasterio.transform import Affine
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from scipy.interpolate import RBFInterpolator

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═════════════════════════════════════════════════════════════
#  Raster I/O
# ═════════════════════════════════════════════════════════════

def load_raster(path, max_dim=4096):
    """Load raster → (PIL.Image, rasterio dataset, downsample factor)."""
    ds = rasterio.open(path)
    h, w = ds.height, ds.width
    factor = max(1, max(h, w) // max_dim)
    out_h, out_w = h // factor, w // factor

    band_count = min(ds.count, 3)
    data = ds.read(list(range(1, band_count + 1)),
                   out_shape=(band_count, out_h, out_w),
                   resampling=Resampling.bilinear)

    img = np.moveaxis(data, 0, -1).astype(np.float64)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    for b in range(img.shape[2]):
        band = img[:, :, b]
        valid = band[band != 0]
        lo, hi = (np.nanpercentile(valid, (2, 98)) if valid.size > 0 else (0, 1))
        if hi - lo < 1e-6:
            hi = lo + 1
        img[:, :, b] = np.clip((band - lo) / (hi - lo) * 255, 0, 255)

    pil_img = Image.fromarray(img.astype(np.uint8), mode="RGB")

    # Display-resolution geo-transform: maps display-pixel → map coords
    disp_transform = ds.transform * Affine.scale(factor, factor)

    return pil_img, ds, factor, disp_transform


def pixel_to_map(ds, col, row):
    x, y = ds.xy(row, col)
    return x, y


def display_to_full(col_d, row_d, factor):
    return col_d * factor, row_d * factor


# ═════════════════════════════════════════════════════════════
#  Geometric Correction Engine
# ═════════════════════════════════════════════════════════════

def compute_corrected_raster(ref_ds, tgt_ds, ref_pts, tgt_pts, output_path,
                             transform_method="auto", progress_cb=None):
    def log(msg):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    n = len(ref_pts)
    log(f"Running geometric correction with {n} GCPs ...")

    ref_map = [pixel_to_map(ref_ds, c, r) for c, r in ref_pts]
    gcps = []
    for i, ((mx, my), (tc, tr)) in enumerate(zip(ref_map, tgt_pts)):
        gcp = GroundControlPoint(row=tr, col=tc, x=mx, y=my, id=str(i))
        gcps.append(gcp)
        log(f"  GCP {i}: tgt px({tc:.1f},{tr:.1f}) -> map({mx:.2f},{my:.2f})")

    if transform_method == "auto":
        if n < 3:
            raise ValueError("Need at least 3 GCPs.")
        method = "affine" if n <= 4 else ("polynomial" if n <= 8 else "tps")
    else:
        method = transform_method
    log(f"  Transform: {method}")

    dst_crs = ref_ds.crs or tgt_ds.crs

    if method in ("affine", "polynomial"):
        dst_tf, dst_w, dst_h = calculate_default_transform(
            dst_crs, dst_crs, ref_ds.width, ref_ds.height, *ref_ds.bounds)
        profile = ref_ds.profile.copy()
        profile.update(driver="GTiff", height=dst_h, width=dst_w,
                       transform=dst_tf, crs=dst_crs,
                       count=tgt_ds.count, dtype=tgt_ds.dtypes[0], compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            for b in range(1, tgt_ds.count + 1):
                log(f"  Reprojecting band {b}/{tgt_ds.count} ...")
                src = tgt_ds.read(b)
                out = np.zeros((dst_h, dst_w), dtype=src.dtype)
                reproject(source=src, destination=out, gcps=gcps,
                          src_crs=dst_crs, dst_transform=dst_tf,
                          dst_crs=dst_crs, resampling=Resampling.bilinear)
                dst.write(out, b)

    elif method == "tps":
        from scipy.ndimage import map_coordinates
        src_pts = np.array(tgt_pts, dtype=np.float64)
        dst_map = np.array(ref_map, dtype=np.float64)
        dst_tf, dst_w, dst_h = calculate_default_transform(
            dst_crs, dst_crs, ref_ds.width, ref_ds.height, *ref_ds.bounds)
        log("  Building TPS inverse mapping ...")
        rbf_c = RBFInterpolator(dst_map, src_pts[:, 0],
                                 kernel="thin_plate_spline", smoothing=0)
        rbf_r = RBFInterpolator(dst_map, src_pts[:, 1],
                                 kernel="thin_plate_spline", smoothing=0)
        co, ro = np.meshgrid(np.arange(dst_w), np.arange(dst_h))
        xs, ys = rasterio.transform.xy(dst_tf, ro.ravel(), co.ravel())
        q = np.column_stack([np.array(xs), np.array(ys)])
        log("  Interpolating ...")
        sc = rbf_c(q).reshape(dst_h, dst_w)
        sr = rbf_r(q).reshape(dst_h, dst_w)
        profile = ref_ds.profile.copy()
        profile.update(driver="GTiff", height=dst_h, width=dst_w,
                       transform=dst_tf, crs=dst_crs,
                       count=tgt_ds.count, dtype=tgt_ds.dtypes[0], compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            for b in range(1, tgt_ds.count + 1):
                log(f"  Warping band {b}/{tgt_ds.count} ...")
                s = tgt_ds.read(b).astype(np.float64)
                w = map_coordinates(s, [sr, sc], order=1, mode="constant", cval=0)
                dst.write(w.astype(tgt_ds.dtypes[0]), b)

    log(f"  Done -> {output_path}  ({dst_w}x{dst_h}, {tgt_ds.count} bands)")
    return output_path


# ═════════════════════════════════════════════════════════════
#  Shared View  (map-coordinate synchronisation)
# ═════════════════════════════════════════════════════════════

class SharedView:
    """Holds a single map-coordinate viewport shared by both panels.

    Attributes:
        cx, cy           : map-coordinate centre of the visible area
        map_per_cpx      : map-units per canvas-pixel (zoom level)
    """

    def __init__(self):
        self.cx = 0.0
        self.cy = 0.0
        self.map_per_cpx = 1.0      # smaller = more zoomed in
        self._listeners = []

    def add_listener(self, fn):
        self._listeners.append(fn)

    def notify(self):
        for fn in self._listeners:
            fn()

    # --- Fit to the union of two bounding boxes ---

    def fit_bounds(self, bounds_list, canvas_w, canvas_h):
        """Set view to fit the union of one or more (left, bottom, right, top) bounds."""
        lefts   = [b[0] for b in bounds_list]
        bottoms = [b[1] for b in bounds_list]
        rights  = [b[2] for b in bounds_list]
        tops    = [b[3] for b in bounds_list]
        x0, x1 = min(lefts), max(rights)
        y0, y1 = min(bottoms), max(tops)
        self.cx = (x0 + x1) / 2
        self.cy = (y0 + y1) / 2
        if canvas_w < 1 or canvas_h < 1:
            return
        sx = (x1 - x0) / canvas_w
        sy = (y1 - y0) / canvas_h
        self.map_per_cpx = max(sx, sy) * 1.05   # 5 % margin
        self.notify()

    # --- Zoom centred on a canvas point ---

    def zoom_at_canvas(self, cx_canvas, cy_canvas, canvas_w, canvas_h, factor):
        """Zoom so the map point under (cx_canvas, cy_canvas) stays fixed."""
        # Map coords under cursor before zoom
        mx = self.cx + (cx_canvas - canvas_w / 2) * self.map_per_cpx
        my = self.cy - (cy_canvas - canvas_h / 2) * self.map_per_cpx  # y flipped

        old_scale = self.map_per_cpx
        self.map_per_cpx = max(1e-12, self.map_per_cpx / factor)

        # Adjust centre so (mx, my) stays under cursor
        self.cx = mx - (cx_canvas - canvas_w / 2) * self.map_per_cpx
        self.cy = my + (cy_canvas - canvas_h / 2) * self.map_per_cpx
        self.notify()

    # --- Pan by canvas pixel delta ---

    def pan_canvas(self, dx_px, dy_px):
        """Pan by a canvas-pixel offset."""
        self.cx -= dx_px * self.map_per_cpx
        self.cy += dy_px * self.map_per_cpx      # y flipped
        self.notify()


# ═════════════════════════════════════════════════════════════
#  Image Panel  (renders one raster in the shared view)
# ═════════════════════════════════════════════════════════════

class ImagePanel(tk.Frame):
    """Canvas that displays a raster in map-coordinate space with GCP overlays."""

    MARKER_R = 7
    LABEL_OFF = 12

    def __init__(self, parent, shared_view: SharedView, title="",
                 border_color="#00cc00", **kw):
        super().__init__(parent, bg="#1e1e1e", **kw)
        self.sv = shared_view
        self.border_color = border_color

        # Title
        tk.Label(self, text=title, bg="#1e1e1e", fg=border_color,
                  font=("Helvetica", 13, "bold")).pack(side=tk.TOP, pady=(4, 0))

        # Canvas
        self.canvas = tk.Canvas(self, bg="#111111", highlightthickness=2,
                                 highlightbackground=border_color, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Coord readout
        self.coord_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.coord_var, bg="#1e1e1e", fg="#888888",
                  font=("Courier", 9)).pack(side=tk.BOTTOM, anchor=tk.W, padx=8)

        # Raster data
        self.pil_image = None
        self.img_w = 0
        self.img_h = 0
        self.disp_transform = None       # Affine: display-pixel → map
        self.inv_transform = None        # Affine: map → display-pixel
        self._photo = None

        # Markers: [(img_x, img_y, index, color, [canvas_ids])]
        self.markers = []

        # Pan state (shift+drag)
        self._pan_start = None

        # --- Binds ---
        self.canvas.bind("<Configure>", lambda e: self.after_idle(self.render))
        self.canvas.bind("<Motion>", self._on_motion)
        # Give canvas focus on mouse-enter (needed for events on macOS)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())

        # NOTE: scroll/zoom is bound at the ROOT level in GCPApp and routed
        #       to the panel under the cursor — not bound here.
        #       ALSO bound directly on canvas as fallback for some tk versions.
        self.canvas.bind("<MouseWheel>", self._on_canvas_scroll)
        self.canvas.bind("<Shift-MouseWheel>", self._on_canvas_shift_scroll)
        # Linux scroll buttons
        self.canvas.bind("<Button-4>", lambda e: self._on_canvas_scroll_linux(1))
        self.canvas.bind("<Button-5>", lambda e: self._on_canvas_scroll_linux(-1))
        self.canvas.bind("<Shift-Button-4>", lambda e: self.zoom_at(
            e.x, e.y, 1.15))
        self.canvas.bind("<Shift-Button-5>", lambda e: self.zoom_at(
            e.x, e.y, 1/1.15))

        # Shift+drag → pan  (ButtonPress-1 with Shift modifier)
        self.canvas.bind("<Shift-ButtonPress-1>", self._pan_press)
        self.canvas.bind("<Shift-B1-Motion>", self._pan_drag)
        self.canvas.bind("<Shift-ButtonRelease-1>", self._pan_release)

        # Also allow right-click/middle-click drag for pan
        for btn in ("2", "3"):
            self.canvas.bind(f"<ButtonPress-{btn}>", self._pan_press)
            self.canvas.bind(f"<B{btn}-Motion>", self._pan_drag)
            self.canvas.bind(f"<ButtonRelease-{btn}>", self._pan_release)

        # Listen for shared view changes
        self.sv.add_listener(self.render)

    # --- Raster setup ---

    def set_raster(self, pil_image, disp_transform):
        self.pil_image = pil_image
        self.img_w, self.img_h = pil_image.size
        self.disp_transform = disp_transform
        self.inv_transform = ~disp_transform    # inverse affine

    # --- Coordinate helpers ---

    def map_to_image(self, mx, my):
        """Map coords → display image pixel (col, row)."""
        col, row = self.inv_transform * (mx, my)
        return col, row

    def image_to_map(self, col, row):
        """Display image pixel → map coords."""
        mx, my = self.disp_transform * (col, row)
        return mx, my

    def canvas_to_map(self, cx, cy):
        """Canvas pixel → map coords via the shared view."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        mx = self.sv.cx + (cx - cw / 2) * self.sv.map_per_cpx
        my = self.sv.cy - (cy - ch / 2) * self.sv.map_per_cpx
        return mx, my

    def map_to_canvas(self, mx, my):
        """Map coords → canvas pixel."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        cx = (mx - self.sv.cx) / self.sv.map_per_cpx + cw / 2
        cy = -(my - self.sv.cy) / self.sv.map_per_cpx + ch / 2
        return cx, cy

    def canvas_to_image_coords(self, cx, cy):
        """Canvas pixel → display image pixel."""
        mx, my = self.canvas_to_map(cx, cy)
        return self.map_to_image(mx, my)

    # --- Rendering ---

    def render(self):
        if self.pil_image is None or self.disp_transform is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        # Visible map extent → image pixel bounds
        mx0, my0 = self.canvas_to_map(0, ch)       # bottom-left in map
        mx1, my1 = self.canvas_to_map(cw, 0)       # top-right in map

        # Image pixel bounds (note: image row increases downward, map y upward)
        ic0, ir0 = self.map_to_image(mx0, my1)     # top-left  image pixel
        ic1, ir1 = self.map_to_image(mx1, my0)     # bottom-right image pixel

        # Clamp to image bounds
        sx0 = max(0, int(np.floor(ic0)))
        sy0 = max(0, int(np.floor(ir0)))
        sx1 = min(self.img_w, int(np.ceil(ic1)))
        sy1 = min(self.img_h, int(np.ceil(ir1)))

        self.canvas.delete("img")

        if sx1 <= sx0 or sy1 <= sy0:
            self._photo = None
            self._redraw_markers()
            return

        crop = self.pil_image.crop((sx0, sy0, sx1, sy1))

        # How many canvas pixels does this crop span?
        tl_cx, tl_cy = self.map_to_canvas(*self.image_to_map(sx0, sy0))
        br_cx, br_cy = self.map_to_canvas(*self.image_to_map(sx1, sy1))
        dw = max(1, int(round(br_cx - tl_cx)))
        dh = max(1, int(round(br_cy - tl_cy)))

        # Determine zoom level for resample method
        img_per_canvas = (sx1 - sx0) / max(dw, 1)
        resample = Image.NEAREST if img_per_canvas < 0.5 else Image.BILINEAR
        disp = crop.resize((dw, dh), resample)

        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(int(tl_cx), int(tl_cy), anchor=tk.NW,
                                  image=self._photo, tags="img")
        self.canvas.tag_lower("img")
        self._redraw_markers()

    # --- Markers ---

    def add_marker(self, img_x, img_y, index, color):
        self.markers.append([img_x, img_y, index, color, []])
        self._draw_marker(len(self.markers) - 1)

    def remove_last_marker(self):
        if not self.markers:
            return
        for cid in self.markers.pop()[4]:
            self.canvas.delete(cid)

    def clear_markers(self):
        for m in self.markers:
            for cid in m[4]:
                self.canvas.delete(cid)
        self.markers.clear()

    def _draw_marker(self, idx):
        ix, iy, label, color, ids = self.markers[idx]
        for cid in ids:
            self.canvas.delete(cid)
        ids.clear()

        # Convert image coords → canvas
        mx, my = self.image_to_map(ix, iy)
        cx, cy = self.map_to_canvas(mx, my)
        r = self.MARKER_R

        ids.append(self.canvas.create_oval(
            cx-r, cy-r, cx+r, cy+r, outline="white", width=2, tags="mk"))
        ids.append(self.canvas.create_oval(
            cx-r+2, cy-r+2, cx+r-2, cy+r-2, outline=color, width=2, tags="mk"))
        ids.append(self.canvas.create_line(
            cx-r-4, cy, cx+r+4, cy, fill=color, width=1, tags="mk"))
        ids.append(self.canvas.create_line(
            cx, cy-r-4, cx, cy+r+4, fill=color, width=1, tags="mk"))
        tid = self.canvas.create_text(
            cx + self.LABEL_OFF, cy - self.LABEL_OFF, text=str(label),
            fill=color, font=("Helvetica", 11, "bold"), anchor=tk.SW, tags="mk")
        bb = self.canvas.bbox(tid)
        if bb:
            bg = self.canvas.create_rectangle(
                bb[0]-2, bb[1]-1, bb[2]+2, bb[3]+1,
                fill="black", outline="", tags="mk")
            self.canvas.tag_lower(bg, tid)
            ids.append(bg)
        ids.append(tid)

    def _redraw_markers(self):
        for i in range(len(self.markers)):
            self._draw_marker(i)

    # --- Zoom (called by GCPApp root-level handler) ---

    def zoom_at(self, canvas_x, canvas_y, factor):
        """Zoom the shared view centred on this panel's canvas coords."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.sv.zoom_at_canvas(canvas_x, canvas_y, cw, ch, factor)

    # --- Canvas-level scroll fallbacks (for tk versions where root binding fails) ---

    def _on_canvas_scroll(self, event):
        """Two-finger scroll on canvas (no shift) → pan."""
        if event.state & 0x0001:
            return  # shift held, skip
        if not hasattr(self, '_scroll_debug'):
            print(f"  [scroll] delta={event.delta}, state={event.state:#x} → pan")
            self._scroll_debug = True
        if platform.system() == "Darwin":
            dy = event.delta * 3
        else:
            dy = event.delta // 4
        self.sv.pan_canvas(0, dy)

    def _on_canvas_shift_scroll(self, event):
        """Shift + scroll on canvas → zoom."""
        if not hasattr(self, '_zoom_debug'):
            print(f"  [shift+scroll] delta={event.delta}, state={event.state:#x} → zoom")
            self._zoom_debug = True
        if platform.system() == "Darwin":
            if event.delta > 0:
                factor = 1.10
            elif event.delta < 0:
                factor = 1 / 1.10
            else:
                return
        else:
            if event.delta > 0:
                factor = 1.15
            elif event.delta < 0:
                factor = 1 / 1.15
            else:
                return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.sv.zoom_at_canvas(event.x, event.y, cw, ch, factor)

    def _on_canvas_scroll_linux(self, direction):
        """Linux scroll (Button-4/5) without shift → pan."""
        self.sv.pan_canvas(0, direction * 40)

    # --- Pan (shift+drag or right/middle drag → updates shared view) ---

    def _pan_press(self, event):
        self._pan_start = (event.x, event.y)

    def _pan_drag(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self.sv.pan_canvas(dx, dy)

    def _pan_release(self, event):
        self._pan_start = None

    # --- Coord readout ---

    def _on_motion(self, event):
        if self.pil_image is None:
            return
        ix, iy = self.canvas_to_image_coords(event.x, event.y)
        mx, my = self.canvas_to_map(event.x, event.y)
        if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
            self.coord_var.set(f"px({ix:.0f},{iy:.0f})  map({mx:.2f},{my:.2f})")
        else:
            self.coord_var.set(f"map({mx:.2f},{my:.2f})")


# ═════════════════════════════════════════════════════════════
#  GCP Table
# ═════════════════════════════════════════════════════════════

class GCPTable(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg="#1e1e1e", **kw)
        cols = ("#", "Ref X", "Ref Y", "Tgt X", "Tgt Y", "Status")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=5,
                                  style="Dark.Treeview")
        widths = [32, 70, 70, 70, 70, 55]
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, minwidth=w, anchor=tk.CENTER)
        sb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def refresh(self, ref_pts, tgt_pts):
        self.tree.delete(*self.tree.get_children())
        for i in range(max(len(ref_pts), len(tgt_pts))):
            rx = f"{ref_pts[i][0]:.0f}" if i < len(ref_pts) else "-"
            ry = f"{ref_pts[i][1]:.0f}" if i < len(ref_pts) else "-"
            tx = f"{tgt_pts[i][0]:.0f}" if i < len(tgt_pts) else "-"
            ty = f"{tgt_pts[i][1]:.0f}" if i < len(tgt_pts) else "-"
            paired = i < len(ref_pts) and i < len(tgt_pts)
            self.tree.insert("", tk.END,
                              values=(i, rx, ry, tx, ty, "Paired" if paired else "..."))


# ═════════════════════════════════════════════════════════════
#  Main Application
# ═════════════════════════════════════════════════════════════

class GCPApp:
    REF_COLOR = "#00ff00"
    TGT_COLOR = "#ff5555"

    def __init__(self, ref_path, tgt_path, output_path, transform_method="auto"):
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.output_path = output_path
        self.transform_method = transform_method

        # GCP data
        self.ref_pts = []           # display image coords
        self.tgt_pts = []
        self.ref_pts_full = []      # full-res pixel coords
        self.tgt_pts_full = []

        self.ref_ds = None
        self.tgt_ds = None
        self.ref_factor = 1
        self.tgt_factor = 1

        # Shared geographic view
        self.shared_view = SharedView()

        self._build_window()
        self._load_rasters()

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("GCP Raster Calibration Tool")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("1440x880")
        self.root.minsize(900, 550)

        # Dark ttk
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview", background="#2a2a2a",
                          foreground="white", fieldbackground="#2a2a2a",
                          font=("Courier", 10))
        style.configure("Dark.Treeview.Heading", background="#333333",
                          foreground="#aaaaaa", font=("Helvetica", 9, "bold"))

        # --- Split panels ---
        panel_frame = tk.Frame(self.root, bg="#1e1e1e")
        panel_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.ref_panel = ImagePanel(panel_frame, self.shared_view,
                                     title="REFERENCE  (base / drone)",
                                     border_color=self.REF_COLOR)
        self.ref_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Frame(panel_frame, bg="#444444", width=2).pack(side=tk.LEFT, fill=tk.Y, padx=1)

        self.tgt_panel = ImagePanel(panel_frame, self.shared_view,
                                     title="TARGET  (to correct)",
                                     border_color=self.TGT_COLOR)
        self.tgt_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # GCP click: left-click WITHOUT Shift
        self.ref_panel.canvas.bind("<ButtonPress-1>", self._on_ref_click)
        self.tgt_panel.canvas.bind("<ButtonPress-1>", self._on_tgt_click)

        # --- Bottom controls ---
        ctrl = tk.Frame(self.root, bg="#282828", height=175)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)
        ctrl.pack_propagate(False)

        self.gcp_table = GCPTable(ctrl)
        self.gcp_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        rc = tk.Frame(ctrl, bg="#282828")
        rc.pack(side=tk.RIGHT, fill=tk.Y, padx=12, pady=6)

        self.counter_var = tk.StringVar(value="GCPs: 0 paired  |  Min 3 required")
        tk.Label(rc, textvariable=self.counter_var, bg="#282828", fg="#00cccc",
                  font=("Helvetica", 12, "bold")).pack(pady=(0, 4))

        self.hint_var = tk.StringVar(value="Next: click REFERENCE panel")
        tk.Label(rc, textvariable=self.hint_var, bg="#282828", fg="#aaaaaa",
                  font=("Helvetica", 10, "italic")).pack(pady=(0, 8))

        tf = tk.Frame(rc, bg="#282828")
        tf.pack(pady=(0, 6))
        tk.Label(tf, text="Transform:", bg="#282828", fg="#aaaaaa",
                  font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.transform_var = tk.StringVar(value=self.transform_method)
        ttk.Combobox(tf, textvariable=self.transform_var, width=12,
                      values=["auto", "affine", "polynomial", "tps"],
                      state="readonly").pack(side=tk.LEFT, padx=6)

        bf = tk.Frame(rc, bg="#282828")
        bf.pack(pady=4)
        tk.Button(bf, text="  Calibrate  ", bg="#2d5a27", fg="white",
                   font=("Helvetica", 12, "bold"), activebackground="#3a7a32",
                   activeforeground="white", command=self._on_calibrate
                   ).grid(row=0, column=0, columnspan=2, pady=(0, 6), sticky="ew")
        tk.Button(bf, text=" Undo ", bg="#555555", fg="white",
                   font=("Helvetica", 10), activebackground="#777777",
                   width=8, command=self._on_undo
                   ).grid(row=1, column=0, padx=2)
        tk.Button(bf, text=" Clear All ", bg="#5a2727", fg="white",
                   font=("Helvetica", 10), activebackground="#7a3232",
                   width=8, command=self._on_clear
                   ).grid(row=1, column=1, padx=2)
        tk.Button(bf, text=" Reset View ", bg="#444444", fg="white",
                   font=("Helvetica", 10), activebackground="#666666",
                   width=18, command=self._on_reset_view
                   ).grid(row=2, column=0, columnspan=2, pady=(6, 0), sticky="ew")

        # Status
        self.status_var = tk.StringVar(
            value="Click: place GCP  |  2-finger scroll: pan  |  "
                  "Shift + scroll: zoom  |  +/-: zoom  |  "
                  "Shift+drag: pan  |  Ctrl+Z: undo  |  H: reset")
        tk.Label(self.root, textvariable=self.status_var, bg="#333333", fg="#aaaaaa",
                  font=("Courier", 9), anchor=tk.W, padx=8
                  ).pack(fill=tk.X, side=tk.BOTTOM)

        # Keys
        self.root.bind("<Control-z>", lambda e: self._on_undo())
        self.root.bind("<Control-Z>", lambda e: self._on_clear())
        self.root.bind("<Control-r>", lambda e: self._on_calibrate())
        self.root.bind("<Control-R>", lambda e: self._on_calibrate())
        self.root.bind("h", lambda e: self._on_reset_view())
        self.root.bind("H", lambda e: self._on_reset_view())

        # Keyboard zoom fallback: +/- or =/- (= is unshifted + on most keyboards)
        self.root.bind("<plus>", lambda e: self._keyboard_zoom(1.25))
        self.root.bind("<equal>", lambda e: self._keyboard_zoom(1.25))
        self.root.bind("<minus>", lambda e: self._keyboard_zoom(1 / 1.25))

        # ── Scroll/zoom is handled at the CANVAS level in each ImagePanel ──
        # (Two-finger scroll = pan, Shift+scroll = zoom, +/- keys = zoom)

    def _keyboard_zoom(self, factor):
        """Zoom from keyboard (+/-) — centred on each panel's middle."""
        for panel in (self.ref_panel, self.tgt_panel):
            cw = panel.canvas.winfo_width()
            ch = panel.canvas.winfo_height()
            panel.zoom_at(cw // 2, ch // 2, factor)

    # --- Load ---

    def _load_rasters(self):
        self.status_var.set("Loading reference raster ...")
        self.root.update_idletasks()
        ref_img, self.ref_ds, self.ref_factor, ref_dtf = load_raster(self.ref_path)
        self.ref_panel.set_raster(ref_img, ref_dtf)

        self.status_var.set("Loading target raster ...")
        self.root.update_idletasks()
        tgt_img, self.tgt_ds, self.tgt_factor, tgt_dtf = load_raster(self.tgt_path)
        self.tgt_panel.set_raster(tgt_img, tgt_dtf)

        self.status_var.set(
            f"Ref: {self.ref_ds.width}x{self.ref_ds.height}  |  "
            f"Tgt: {self.tgt_ds.width}x{self.tgt_ds.height}  |  Ready")

        print(f"\n{'='*60}")
        print(f"  Reference : {self.ref_path}")
        print(f"    {self.ref_ds.width}x{self.ref_ds.height}, "
              f"{self.ref_ds.count} bands, CRS={self.ref_ds.crs}")
        print(f"    Bounds  : {self.ref_ds.bounds}")
        print(f"  Target    : {self.tgt_path}")
        print(f"    {self.tgt_ds.width}x{self.tgt_ds.height}, "
              f"{self.tgt_ds.count} bands, CRS={self.tgt_ds.crs}")
        print(f"    Bounds  : {self.tgt_ds.bounds}")
        print(f"  Output    : {self.output_path}")
        print(f"{'='*60}\n")

    # --- GCP clicks (only when Shift is NOT held) ---

    def _is_shift(self, event):
        return bool(event.state & 0x0001)

    def _on_ref_click(self, event):
        if self._is_shift(event):
            return      # shift+click = pan, handled by panel
        ix, iy = self.ref_panel.canvas_to_image_coords(event.x, event.y)
        if not (0 <= ix < self.ref_panel.img_w and 0 <= iy < self.ref_panel.img_h):
            return

        idx = len(self.ref_pts)
        self.ref_pts.append((ix, iy))
        fc, fr = display_to_full(ix, iy, self.ref_factor)
        self.ref_pts_full.append((fc, fr))
        self.ref_panel.add_marker(ix, iy, idx, self.REF_COLOR)

        mx, my = pixel_to_map(self.ref_ds, fc, fr)
        print(f"  REF {idx}: img({ix:.0f},{iy:.0f})  px({fc:.0f},{fr:.0f})  "
              f"map({mx:.2f},{my:.2f})")
        self._update_ui()

    def _on_tgt_click(self, event):
        if self._is_shift(event):
            return
        ix, iy = self.tgt_panel.canvas_to_image_coords(event.x, event.y)
        if not (0 <= ix < self.tgt_panel.img_w and 0 <= iy < self.tgt_panel.img_h):
            return

        idx = len(self.tgt_pts)
        self.tgt_pts.append((ix, iy))
        fc, fr = display_to_full(ix, iy, self.tgt_factor)
        self.tgt_pts_full.append((fc, fr))
        self.tgt_panel.add_marker(ix, iy, idx, self.TGT_COLOR)

        print(f"  TGT {idx}: img({ix:.0f},{iy:.0f})  px({fc:.0f},{fr:.0f})")
        self._update_ui()

    def _update_ui(self):
        nr, nt = len(self.ref_pts), len(self.tgt_pts)
        paired = min(nr, nt)
        need = max(0, 3 - paired)
        tag = "Ready to calibrate" if paired >= 3 else f"Need {need} more pair(s)"
        self.counter_var.set(f"GCPs: {paired} paired  (R:{nr}  T:{nt})  |  {tag}")
        if nr <= nt:
            self.hint_var.set("Next: click REFERENCE panel (green)")
        else:
            self.hint_var.set("Next: click TARGET panel (red)")
        self.gcp_table.refresh(self.ref_pts, self.tgt_pts)

    # --- Actions ---

    def _on_undo(self):
        nr, nt = len(self.ref_pts), len(self.tgt_pts)
        if nr == 0 and nt == 0:
            return
        if nt >= nr and nt > 0:
            self.tgt_pts.pop(); self.tgt_pts_full.pop()
            self.tgt_panel.remove_last_marker()
            print("  Undid last TGT point")
        elif nr > 0:
            self.ref_pts.pop(); self.ref_pts_full.pop()
            self.ref_panel.remove_last_marker()
            print("  Undid last REF point")
        self._update_ui()

    def _on_clear(self):
        self.ref_pts.clear(); self.tgt_pts.clear()
        self.ref_pts_full.clear(); self.tgt_pts_full.clear()
        self.ref_panel.clear_markers()
        self.tgt_panel.clear_markers()
        print("  All points cleared")
        self._update_ui()

    def _on_reset_view(self):
        bounds = []
        if self.ref_ds:
            bounds.append(self.ref_ds.bounds)
        if self.tgt_ds:
            bounds.append(self.tgt_ds.bounds)
        if not bounds:
            return
        cw = max(self.ref_panel.canvas.winfo_width(), 100)
        ch = max(self.ref_panel.canvas.winfo_height(), 100)
        self.shared_view.fit_bounds(bounds, cw, ch)

    def _on_calibrate(self):
        paired = min(len(self.ref_pts_full), len(self.tgt_pts_full))
        if paired < 3:
            messagebox.showwarning("Not enough GCPs",
                                    f"Need at least 3 paired GCPs.\nCurrently have {paired}.")
            return
        rp = self.ref_pts_full[:paired]
        tp = self.tgt_pts_full[:paired]
        method = self.transform_var.get()
        self.status_var.set("Calibrating ... please wait")
        self.root.update_idletasks()

        def work():
            try:
                compute_corrected_raster(
                    self.ref_ds, self.tgt_ds, rp, tp,
                    self.output_path, transform_method=method,
                    progress_cb=lambda m: self.root.after(
                        0, lambda msg=m: self.status_var.set(msg)))
                self.root.after(0, lambda: self.status_var.set(
                    f"Done -> {self.output_path}"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Calibration Complete",
                    f"Corrected raster saved:\n{self.output_path}"))
            except Exception as e:
                print(f"  Error: {e}")
                self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def run(self):
        self.root.after(200, self._on_reset_view)
        self.root.mainloop()


# ═════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GCP-based raster calibration -- synchronised split-screen tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gcp_calibrator.py drone_ortho.tif satellite.tif
  python gcp_calibrator.py base.tif target.tif --output aligned.tif
  python gcp_calibrator.py base.tif target.tif -t polynomial

Packaging:
  pip install pyinstaller
  pyinstaller --onefile --windowed gcp_calibrator.py
        """)
    parser.add_argument("reference", help="Reference (base) raster")
    parser.add_argument("target", help="Target raster to correct")
    parser.add_argument("-o", "--output", default=None,
                         help="Output path (default: <target>_corrected.tif)")
    parser.add_argument("-t", "--transform",
                         choices=["auto", "affine", "polynomial", "tps"],
                         default="auto", help="Transform method")
    args = parser.parse_args()

    for p in (args.reference, args.target):
        if not Path(p).is_file():
            print(f"Error: not found -> {p}")
            sys.exit(1)

    if args.output is None:
        stem = Path(args.target).stem
        sfx = Path(args.target).suffix or ".tif"
        args.output = str(Path(args.target).parent / f"{stem}_corrected{sfx}")

    app = GCPApp(args.reference, args.target, args.output, args.transform)
    app.run()


if __name__ == "__main__":
    main()