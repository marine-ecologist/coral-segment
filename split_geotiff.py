#!/usr/bin/env python3
"""
split_geotiff.py - Split GeoTIFF into separate georeferenced tiles by metre size

Creates individual GeoTIFF files for each tile, maintaining proper georeferencing.
Optional buffer creates overlap between tiles.

Usage:
    python split_geotiff.py input.tif output_folder/ --tile-size 50
    python split_geotiff.py input.tif output_folder/ --tile-size 50 --buffer 2
    python split_geotiff.py input.tif output_folder/ --tile-size 100 --format GTiff
"""

import sys
import numpy as np
from pathlib import Path

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
except ImportError:
    print("Installing rasterio...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterio", "--break-system-packages"])
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds


def split_geotiff_by_meters(input_path, output_folder, tile_size_meters=50, 
                            buffer_meters=0, driver='GTiff', compress='LZW'):
    """
    Split a GeoTIFF into separate tiles by metre size
    
    Args:
        input_path: Path to input GeoTIFF
        output_folder: Output folder for tile GeoTIFFs
        tile_size_meters: Tile size in metres (creates square tiles)
        buffer_meters: Buffer around each tile in metres (creates overlap)
        driver: Output format (GTiff, COG, etc.)
        compress: Compression method (LZW, DEFLATE, JPEG, etc.)
    
    Returns:
        List of created tile paths
    """
    output_folder = Path(output_folder)
    try:
        output_folder.mkdir(exist_ok=True, parents=True, mode=0o755)
        # Ensure we have write permissions
        import os
        os.chmod(output_folder, 0o755)
    except PermissionError:
        print(f"Error: No permission to create folder: {output_folder}")
        print(f"Try running: sudo mkdir -p \"{output_folder}\" && sudo chmod 755 \"{output_folder}\"")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating output folder: {e}")
        sys.exit(1)
    
    # Verify we can write to the folder
    test_file = output_folder / '.write_test'
    try:
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        print(f"Error: Cannot write to folder: {output_folder}")
        print(f"Try running: sudo chmod 755 \"{output_folder}\"")
        sys.exit(1)
    
    print(f"Opening GeoTIFF: {input_path}")
    
    with rasterio.open(input_path) as src:
        # Get image info
        width = src.width
        height = src.height
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        pixel_res_x, pixel_res_y = src.res
        
        print(f"\nImage Information:")
        print(f"  Size: {width} x {height} pixels")
        print(f"  CRS: {crs}")
        print(f"  Bounds: {bounds}")
        print(f"  Resolution: {pixel_res_x:.6f} x {pixel_res_y:.6f} units per pixel")
        print(f"  Bands: {src.count}")
        
        # Check if CRS is projected
        if not crs.is_projected:
            print(f"\n⚠ WARNING: CRS is geographic (lat/lon), not projected.")
            print(f"  Metre-based measurements may not be accurate.")
            print(f"  For accurate results, reproject to a metric CRS (e.g., UTM).")
            print(f"  Assuming 1 degree ≈ 111,000 metres (rough approximation)")
            pixel_res_x = pixel_res_x * 111000
            pixel_res_y = pixel_res_y * 111000
        else:
            print(f"  CRS is projected - metre calculations will be accurate")
        
        # Convert tile size from metres to pixels
        tile_width_px = int(tile_size_meters / abs(pixel_res_x))
        tile_height_px = int(tile_size_meters / abs(pixel_res_y))
        
        # Convert buffer from metres to pixels
        buffer_px_x = int(buffer_meters / abs(pixel_res_x))
        buffer_px_y = int(buffer_meters / abs(pixel_res_y))
        
        print(f"\nTiling Configuration:")
        print(f"  Tile size: {tile_size_meters}m x {tile_size_meters}m")
        print(f"  Tile size in pixels: {tile_width_px} x {tile_height_px}")
        if buffer_meters > 0:
            print(f"  Buffer: {buffer_meters}m ({buffer_px_x} x {buffer_px_y} pixels)")
            print(f"  Actual tile size with buffer: {tile_size_meters + 2*buffer_meters}m x {tile_size_meters + 2*buffer_meters}m")
            print(f"  Actual tile size in pixels: {tile_width_px + 2*buffer_px_x} x {tile_height_px + 2*buffer_px_y}")
        
        # Calculate number of tiles (based on nominal size, not buffered)
        n_tiles_x = int(np.ceil(width / tile_width_px))
        n_tiles_y = int(np.ceil(height / tile_height_px))
        total_tiles = n_tiles_x * n_tiles_y
        
        print(f"  Grid: {n_tiles_x} x {n_tiles_y}")
        print(f"  Total tiles: {total_tiles}")
        
        # Get data type and nodata value
        dtype = src.dtypes[0]
        nodata = src.nodata
        
        print(f"\nOutput Configuration:")
        print(f"  Format: {driver}")
        print(f"  Compression: {compress}")
        print(f"  Data type: {dtype}")
        
        # Create tiles
        print(f"\nCreating tile GeoTIFFs...")
        tile_count = 0
        created_files = []
        
        for tile_y in range(n_tiles_y):
            for tile_x in range(n_tiles_x):
                # Calculate nominal window position (center of tile, no buffer)
                nominal_col_off = tile_x * tile_width_px
                nominal_row_off = tile_y * tile_height_px
                
                # Calculate buffered window position (extend in all directions)
                col_off = max(0, nominal_col_off - buffer_px_x)
                row_off = max(0, nominal_row_off - buffer_px_y)
                
                # Calculate buffered dimensions
                # Right edge: nominal position + tile width + buffer, but don't exceed image
                col_end = min(nominal_col_off + tile_width_px + buffer_px_x, width)
                row_end = min(nominal_row_off + tile_height_px + buffer_px_y, height)
                
                actual_width = col_end - col_off
                actual_height = row_end - row_off
                
                if actual_width <= 0 or actual_height <= 0:
                    continue
                
                window = Window(col_off, row_off, actual_width, actual_height)
                
                # Read tile data (all bands)
                tile_data = src.read(window=window)
                
                # Calculate transform for this tile
                window_transform = src.window_transform(window)
                
                # Calculate geographic bounds for this tile
                x_min, y_max = window_transform * (0, 0)
                x_max, y_min = window_transform * (actual_width, actual_height)
                
                # Generate filename
                tile_name = f"tile_{tile_y:04d}_{tile_x:04d}.tif"
                tile_path = output_folder / tile_name
                
                # Save tile as GeoTIFF
                with rasterio.open(
                    tile_path,
                    'w',
                    driver=driver,
                    height=actual_height,
                    width=actual_width,
                    count=src.count,
                    dtype=dtype,
                    crs=crs,
                    transform=window_transform,
                    nodata=nodata,
                    compress=compress
                ) as dst:
                    dst.write(tile_data)
                    
                    # Copy band descriptions if they exist
                    for i in range(1, src.count + 1):
                        desc = src.descriptions[i-1]
                        if desc:
                            dst.set_band_description(i, desc)
                
                created_files.append(tile_path)
                
                tile_count += 1
                if tile_count % 100 == 0 or tile_count == total_tiles:
                    print(f"  Created {tile_count}/{total_tiles} tiles")
        
        print(f"\n✓ Created {tile_count} tile GeoTIFFs")
    
    # Create index file
    index_path = output_folder / 'tile_index.txt'
    with open(index_path, 'w') as f:
        f.write("filename,tile_x,tile_y,width_px,height_px,width_m,height_m,"
               "nominal_width_m,nominal_height_m,buffer_m,"
               "bounds_left,bounds_bottom,bounds_right,bounds_top\n")
        
        tile_idx = 0
        for tile_y in range(n_tiles_y):
            for tile_x in range(n_tiles_x):
                nominal_col_off = tile_x * tile_width_px
                nominal_row_off = tile_y * tile_height_px
                
                col_off = max(0, nominal_col_off - buffer_px_x)
                row_off = max(0, nominal_row_off - buffer_px_y)
                
                col_end = min(nominal_col_off + tile_width_px + buffer_px_x, width)
                row_end = min(nominal_row_off + tile_height_px + buffer_px_y, height)
                
                actual_width = col_end - col_off
                actual_height = row_end - row_off
                
                if actual_width <= 0 or actual_height <= 0:
                    continue
                
                if tile_idx < len(created_files):
                    tile_path = created_files[tile_idx]
                    
                    # Get bounds from the saved file
                    with rasterio.open(tile_path) as tile_src:
                        tile_bounds = tile_src.bounds
                        
                        width_m = tile_bounds.right - tile_bounds.left
                        height_m = tile_bounds.top - tile_bounds.bottom
                        nominal_width_m = tile_size_meters
                        nominal_height_m = tile_size_meters
                        
                        f.write(f"{tile_path.name},{tile_x},{tile_y},"
                               f"{actual_width},{actual_height},"
                               f"{width_m:.2f},{height_m:.2f},"
                               f"{nominal_width_m:.2f},{nominal_height_m:.2f},"
                               f"{buffer_meters:.2f},"
                               f"{tile_bounds.left:.6f},{tile_bounds.bottom:.6f},"
                               f"{tile_bounds.right:.6f},{tile_bounds.top:.6f}\n")
                    
                    tile_idx += 1
    
    print(f"✓ Saved tile index: {index_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SPLITTING COMPLETE")
    print(f"{'='*70}")
    print(f"Input: {input_path}")
    print(f"Output folder: {output_folder}")
    print(f"Tile size: {tile_size_meters}m x {tile_size_meters}m ({tile_width_px}x{tile_height_px} pixels)")
    if buffer_meters > 0:
        print(f"Buffer: {buffer_meters}m ({buffer_px_x}x{buffer_px_y} pixels)")
        print(f"Actual tile size: {tile_size_meters + 2*buffer_meters}m x {tile_size_meters + 2*buffer_meters}m")
        print(f"Overlap: Tiles overlap by {2*buffer_meters}m")
    print(f"Grid: {n_tiles_x} x {n_tiles_y}")
    print(f"Total tiles: {tile_count}")
    print(f"Format: {driver} (compression: {compress})")
    print(f"Index: {index_path}")
    print(f"{'='*70}")
    
    return created_files


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nError: Missing arguments")
        print("\nExamples:")
        print("  # Split into 50m x 50m tiles")
        print("  python split_geotiff_by_meters.py input.tif output_tiles/ --tile-size 50")
        print("\n  # Split into 50m x 50m tiles with 2m buffer (54m x 54m actual)")
        print("  python split_geotiff_by_meters.py input.tif output_tiles/ --tile-size 50 --buffer 2")
        print("\n  # Split into 100m x 100m tiles with 5m buffer")
        print("  python split_geotiff_by_meters.py input.tif output_tiles/ --tile-size 100 --buffer 5")
        print("\n  # Use different compression")
        print("  python split_geotiff_by_meters.py input.tif output_tiles/ --tile-size 50 --compress DEFLATE")
        print("\n  # Create Cloud Optimized GeoTIFFs (COG)")
        print("  python split_geotiff_by_meters.py input.tif output_tiles/ --tile-size 50 --format COG")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Parse arguments
    tile_size_meters = 50  # Default 50m
    buffer_meters = 0  # Default no buffer
    driver = 'GTiff'
    compress = 'LZW'
    
    if '--tile-size' in sys.argv:
        idx = sys.argv.index('--tile-size')
        tile_size_meters = float(sys.argv[idx + 1])
    
    if '--buffer' in sys.argv:
        idx = sys.argv.index('--buffer')
        buffer_meters = float(sys.argv[idx + 1])
    
    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        driver = sys.argv[idx + 1]
        if driver not in ['GTiff', 'COG']:
            print(f"Warning: Unusual format '{driver}'. Common options: GTiff, COG")
    
    if '--compress' in sys.argv:
        idx = sys.argv.index('--compress')
        compress = sys.argv[idx + 1]
        if compress.upper() not in ['LZW', 'DEFLATE', 'JPEG', 'PACKBITS', 'NONE']:
            print(f"Warning: Unusual compression '{compress}'. Common options: LZW, DEFLATE, JPEG, PACKBITS, NONE")
    
    try:
        created_files = split_geotiff_by_meters(
            input_path,
            output_folder,
            tile_size_meters=tile_size_meters,
            buffer_meters=buffer_meters,
            driver=driver,
            compress=compress
        )
        print(f"\n✓ Success! Created {len(created_files)} tile files")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()