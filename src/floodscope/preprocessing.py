"""
FloodScope AI - Preprocessing and Spatial Alignment Module
Member 2: Detection (Image Preprocessing, Metadata Inspection & Spatial Alignment)

This module handles:
1. Safe loading of multi-sensor satellite imagery (RGB optical, multi-band, and SAR).
2. Extraction and validation of GeoTIFF spatial metadata (CRS, tiepoints, pixel scale, bounds).
3. Geospatial compatibility checking (coordinate systems, footprint overlap, resolution).
4. Sub-pixel accurate spatial alignment via rigorous geospatial affine transformations.
5. Generation of alignment visual diagnostics (checkerboard, overlay, side-by-side).
6. Machine-readable audit reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tifffile


class SpatialAlignmentError(Exception):
    """Raised when images cannot be reliably aligned or geospatial metadata is incompatible."""
    pass


def load_image(file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Safely loads a satellite raster image, returning the image array and basic metadata.

    Supports:
    - 3-band RGB/BGR optical GeoTIFFs and common image formats.
    - Multi-band optical rasters (e.g. 13-band Sentinel-2 L1C).
    - Multi-polarization SAR rasters (e.g. 2-band Sentinel-1 float32).

    Parameters:
        file_path: Path to the raster file.

    Returns:
        Tuple of (numpy array, metadata dict)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path.resolve()}")

    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): {path.resolve()}")

    meta: Dict[str, Any] = {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "loader": None,
        "channels": 0,
        "height": 0,
        "width": 0,
        "dtype": None,
        "min_val": None,
        "max_val": None,
    }

    try:
        # Tifffile handles arbitrary band counts, float32, and GeoTIFF tags natively
        arr = tifffile.imread(str(path))
        meta["loader"] = "tifffile"
    except Exception as tiff_err:
        # Fall back to OpenCV for standard image formats
        try:
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"cv2.imread failed to decode {path.name}")
            meta["loader"] = "opencv"
        except Exception as cv_err:
            raise ValueError(f"Failed to read image '{path.name}'. Tifffile error: {tiff_err}; OpenCV error: {cv_err}")

    # Inspect dimensions and channel order
    # Multi-band rasters from tifffile often come as (channels, height, width)
    if arr.ndim == 2:
        h, w = arr.shape
        c = 1
    elif arr.ndim == 3:
        if arr.shape[0] in [1, 2, 4, 12, 13] and arr.shape[1] > 32 and arr.shape[2] > 32:
            # (channels, height, width) -> reorder to (height, width, channels)
            c, h, w = arr.shape
            arr = np.transpose(arr, (1, 2, 0))
        else:
            h, w, c = arr.shape
    else:
        raise ValueError(f"Unsupported image array dimensions: {arr.shape}")

    meta["height"] = int(h)
    meta["width"] = int(w)
    meta["channels"] = int(c)
    meta["dtype"] = str(arr.dtype)

    # Compute valid min/max
    finite_mask = np.isfinite(arr)
    if np.any(finite_mask):
        meta["min_val"] = float(np.min(arr[finite_mask]))
        meta["max_val"] = float(np.max(arr[finite_mask]))

    return arr, meta


def extract_geotiff_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts geographic metadata (CRS, tiepoints, pixel scale, geographic bounding box)
    from a GeoTIFF using tifffile.

    Parameters:
        file_path: Path to the GeoTIFF file.

    Returns:
        Dict containing georeferencing information and bounding box coordinates.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    meta: Dict[str, Any] = {
        "file_name": path.name,
        "has_georeference": False,
        "crs": None,
        "crs_epsg": None,
        "tiepoint": None,
        "pixel_scale": None,
        "bounds": None,
        "width": 0,
        "height": 0,
        "pixel_res_deg": None,
        "estimated_res_m": None,
    }

    try:
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            meta["width"] = int(page.imagewidth)
            meta["height"] = int(page.imagelength)

            geotiff_dict = getattr(tf, "geotiff_metadata", None)
            if geotiff_dict is None:
                return meta

            # Extract CRS
            epsg = geotiff_dict.get("GeographicTypeGeoKey")
            if epsg is not None:
                # Handle enum or int
                epsg_val = int(getattr(epsg, "value", epsg))
                meta["crs_epsg"] = epsg_val
                meta["crs"] = f"EPSG:{epsg_val}"
            elif "GTModelTypeGeoKey" in geotiff_dict:
                meta["crs"] = str(geotiff_dict.get("GTModelTypeGeoKey"))

            # Extract tiepoints and scale
            tiepoint = geotiff_dict.get("ModelTiepoint")
            pixel_scale = geotiff_dict.get("ModelPixelScale")

            if tiepoint is not None and pixel_scale is not None and len(tiepoint) >= 6 and len(pixel_scale) >= 2:
                meta["has_georeference"] = True
                meta["tiepoint"] = [float(v) for v in tiepoint]
                meta["pixel_scale"] = [float(v) for v in pixel_scale]

                # In standard north-up GeoTIFF:
                # tiepoint[3] = upper-left longitude (x_origin)
                # tiepoint[4] = upper-left latitude (y_origin)
                # pixel_scale[0] = delta lon per pixel
                # pixel_scale[1] = delta lat per pixel
                x_origin = meta["tiepoint"][3]
                y_origin = meta["tiepoint"][4]
                sx = meta["pixel_scale"][0]
                sy = meta["pixel_scale"][1]

                lon_min = x_origin
                lon_max = x_origin + meta["width"] * sx
                lat_max = y_origin
                lat_min = y_origin - meta["height"] * sy

                meta["bounds"] = {
                    "lon_min": float(lon_min),
                    "lon_max": float(lon_max),
                    "lat_min": float(lat_min),
                    "lat_max": float(lat_max),
                }

                meta["pixel_res_deg"] = (float(sx), float(sy))
                # Approximate 1 degree latitude ~ 111,320 meters
                # 1 degree longitude ~ 111,320 * cos(mid_lat)
                mid_lat = (lat_min + lat_max) / 2.0
                m_per_deg_lat = 111320.0
                m_per_deg_lon = 111320.0 * np.cos(np.radians(mid_lat))
                res_x_m = sx * m_per_deg_lon
                res_y_m = sy * m_per_deg_lat
                meta["estimated_res_m"] = (round(float(res_x_m), 3), round(float(res_y_m), 3))

    except Exception as e:
        meta["error"] = str(e)

    return meta


def evaluate_spatial_compatibility(
    meta_pre: Dict[str, Any],
    meta_post: Dict[str, Any],
    min_overlap_pct: float = 80.0
) -> Dict[str, Any]:
    """
    Evaluates whether two GeoTIFF scenes are compatible for spatial alignment
    based on CRS, bounding box overlap, and coordinate bounds.

    Parameters:
        meta_pre: Metadata dict of PRE image from extract_geotiff_metadata().
        meta_post: Metadata dict of POST image from extract_geotiff_metadata().
        min_overlap_pct: Minimum geographic area overlap percentage required.

    Returns:
        Dict with compatibility verdict, overlap percentages, and diagnostic warnings.
    """
    comp: Dict[str, Any] = {
        "is_compatible": False,
        "crs_matched": False,
        "overlap_pct_pre": 0.0,
        "overlap_pct_post": 0.0,
        "intersection_bounds": None,
        "tiepoint_offset_px": None,
        "scale_ratio": None,
        "warnings": [],
    }

    if not meta_pre.get("has_georeference"):
        comp["warnings"].append("PRE image lacks valid GeoTIFF georeferencing metadata.")
        return comp

    if not meta_post.get("has_georeference"):
        comp["warnings"].append("POST image lacks valid GeoTIFF georeferencing metadata.")
        return comp

    # Check CRS
    crs_pre = meta_pre.get("crs")
    crs_post = meta_post.get("crs")
    if crs_pre != crs_post or crs_pre is None:
        comp["warnings"].append(f"CRS mismatch: PRE={crs_pre} vs POST={crs_post}")
        return comp

    comp["crs_matched"] = True

    b_pre = meta_pre["bounds"]
    b_post = meta_post["bounds"]

    # Calculate geographic intersection
    inter_lon_min = max(b_pre["lon_min"], b_post["lon_min"])
    inter_lon_max = min(b_pre["lon_max"], b_post["lon_max"])
    inter_lat_min = max(b_pre["lat_min"], b_post["lat_min"])
    inter_lat_max = min(b_pre["lat_max"], b_post["lat_max"])

    inter_width = max(0.0, inter_lon_max - inter_lon_min)
    inter_height = max(0.0, inter_lat_max - inter_lat_min)
    inter_area = inter_width * inter_height

    pre_area = (b_pre["lon_max"] - b_pre["lon_min"]) * (b_pre["lat_max"] - b_pre["lat_min"])
    post_area = (b_post["lon_max"] - b_post["lon_min"]) * (b_post["lat_max"] - b_post["lat_min"])

    pct_pre = (inter_area / pre_area) * 100.0 if pre_area > 0 else 0.0
    pct_post = (inter_area / post_area) * 100.0 if post_area > 0 else 0.0

    comp["overlap_pct_pre"] = round(float(pct_pre), 4)
    comp["overlap_pct_post"] = round(float(pct_post), 4)

    if inter_area <= 0:
        comp["warnings"].append("Images have zero geographic overlap (completely disjoint scenes).")
        return comp

    comp["intersection_bounds"] = {
        "lon_min": float(inter_lon_min),
        "lon_max": float(inter_lon_max),
        "lat_min": float(inter_lat_min),
        "lat_max": float(inter_lat_max),
    }

    # Relative pixel scale ratio (PRE / POST)
    scale_pre = meta_pre["pixel_scale"]
    scale_post = meta_post["pixel_scale"]
    scale_ratio_x = scale_pre[0] / scale_post[0]
    scale_ratio_y = scale_pre[1] / scale_post[1]
    comp["scale_ratio"] = (round(float(scale_ratio_x), 6), round(float(scale_ratio_y), 6))

    # Offset of PRE upper-left tiepoint relative to POST upper-left tiepoint in POST pixels
    tie_pre = meta_pre["tiepoint"]
    tie_post = meta_post["tiepoint"]
    offset_x_post_px = (tie_pre[3] - tie_post[3]) / scale_post[0]
    offset_y_post_px = (tie_post[4] - tie_pre[4]) / scale_post[1]
    comp["tiepoint_offset_px"] = (round(float(offset_x_post_px), 4), round(float(offset_y_post_px), 4))

    if pct_pre < min_overlap_pct or pct_post < min_overlap_pct:
        comp["warnings"].append(
            f"Overlap below threshold ({min_overlap_pct}%): PRE overlap={pct_pre:.2f}%, POST overlap={pct_post:.2f}%"
        )
        return comp

    comp["is_compatible"] = True
    return comp


def compute_geospatial_affine_matrix(
    meta_src: Dict[str, Any],
    meta_dst: Dict[str, Any]
) -> np.ndarray:
    """
    Computes the exact 2x3 forward affine transformation matrix M mapping source
    image coordinates (x_src, y_src) to destination grid coordinates (x_dst, y_dst):

        [x_dst, y_dst]^T = M * [x_src, y_src, 1]^T

    When passed to cv2.warpAffine(img_src, M, (dst_w, dst_h)), OpenCV resamples
    img_src onto the destination spatial grid using standard forward convention.

    Parameters:
        meta_src: Metadata dict of the source image to be warped.
        meta_dst: Metadata dict of the destination (reference) grid.

    Returns:
        2x3 float64 affine transformation matrix.
    """
    tie_src = meta_src["tiepoint"]
    scale_src = meta_src["pixel_scale"]
    tie_dst = meta_dst["tiepoint"]
    scale_dst = meta_dst["pixel_scale"]

    # For any source coordinate (x_src, y_src):
    # lon = tie_src[3] + x_src * scale_src[0]
    # lat = tie_src[4] - y_src * scale_src[1]
    #
    # In destination grid:
    # x_dst = (lon - tie_dst[3]) / scale_dst[0]
    #       = (scale_src[0] / scale_dst[0]) * x_src + (tie_src[3] - tie_dst[3]) / scale_dst[0]
    #
    # y_dst = (tie_dst[4] - lat) / scale_dst[1]
    #       = (scale_src[1] / scale_dst[1]) * y_src + (tie_dst[4] - tie_src[4]) / scale_dst[1]

    m00 = scale_src[0] / scale_dst[0]
    m01 = 0.0
    m02 = (tie_src[3] - tie_dst[3]) / scale_dst[0]

    m10 = 0.0
    m11 = scale_src[1] / scale_dst[1]
    m12 = (tie_dst[4] - tie_src[4]) / scale_dst[1]

    return np.array([
        [m00, m01, m02],
        [m10, m11, m12]
    ], dtype=np.float64)


def align_image_pair(
    pre_path: Union[str, Path],
    post_path: Union[str, Path],
    target_grid: str = "pre",
    interpolation: int = cv2.INTER_LINEAR
) -> Dict[str, Any]:
    """
    Loads, checks compatibility, and resamples PRE and POST satellite images onto
    a unified, common spatial grid using exact geospatial metadata.

    Parameters:
        pre_path: Path to pre-event image.
        post_path: Path to post-event image.
        target_grid: Reference spatial grid ('pre' or 'post'). Defaults to 'pre'.
        interpolation: OpenCV interpolation flag (e.g. cv2.INTER_LINEAR, cv2.INTER_NEAREST).

    Returns:
        Dict containing:
        - status: 'SUCCESS' or 'FAILED'
        - pre_aligned: np.ndarray on common grid
        - post_aligned: np.ndarray on common grid
        - overlap_mask: binary uint8 mask (255 where both images have valid geographic data)
        - metadata: complete alignment record
        - warnings: list of diagnostic notes or warnings
    """
    pre_p = Path(pre_path)
    post_p = Path(post_path)

    # 1. Load images safely
    img_pre, info_pre = load_image(pre_p)
    img_post, info_post = load_image(post_p)

    # 2. Extract geospatial metadata
    meta_pre = extract_geotiff_metadata(pre_p)
    meta_post = extract_geotiff_metadata(post_p)

    # 3. Evaluate compatibility
    comp = evaluate_spatial_compatibility(meta_pre, meta_post)
    if not comp["is_compatible"]:
        raise SpatialAlignmentError(
            f"Images cannot be aligned: {'; '.join(comp['warnings'])}"
        )

    # 4. Perform geospatial resampling onto common grid
    if target_grid.lower() == "pre":
        ref_h, ref_w = meta_pre["height"], meta_pre["width"]
        M = compute_geospatial_affine_matrix(meta_src=meta_post, meta_dst=meta_pre)
        
        post_aligned = cv2.warpAffine(
            img_post, M, (ref_w, ref_h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        pre_aligned = img_pre.copy()
        
        # Valid footprint mask for post in pre grid
        mask_ones = np.ones((meta_post["height"], meta_post["width"]), dtype=np.uint8) * 255
        post_valid_mask = cv2.warpAffine(
            mask_ones, M, (ref_w, ref_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        pre_valid_mask = np.ones((ref_h, ref_w), dtype=np.uint8) * 255
        overlap_mask = cv2.bitwise_and(pre_valid_mask, post_valid_mask)
        
    elif target_grid.lower() == "post":
        ref_h, ref_w = meta_post["height"], meta_post["width"]
        M = compute_geospatial_affine_matrix(meta_src=meta_pre, meta_dst=meta_post)
        
        pre_aligned = cv2.warpAffine(
            img_pre, M, (ref_w, ref_h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        post_aligned = img_post.copy()
        
        mask_ones = np.ones((meta_pre["height"], meta_pre["width"]), dtype=np.uint8) * 255
        pre_valid_mask = cv2.warpAffine(
            mask_ones, M, (ref_w, ref_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        post_valid_mask = np.ones((ref_h, ref_w), dtype=np.uint8) * 255
        overlap_mask = cv2.bitwise_and(pre_valid_mask, post_valid_mask)
    else:
        raise ValueError(f"Unknown target_grid '{target_grid}'. Must be 'pre' or 'post'.")

    # Compute valid overlap pixel ratio
    total_pixels = ref_h * ref_w
    valid_pixels = int(np.count_nonzero(overlap_mask))
    valid_ratio = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0

    alignment_metadata = {
        "status": "SUCCESS",
        "target_grid": target_grid,
        "grid_dimensions": (ref_h, ref_w),
        "affine_matrix": M.tolist(),
        "valid_overlap_pixels": valid_pixels,
        "valid_overlap_ratio": round(valid_ratio, 6),
        "compatibility": comp,
        "pre_info": info_pre,
        "post_info": info_post,
        "meta_pre": meta_pre,
        "meta_post": meta_post,
        "warnings": comp["warnings"],
    }

    return {
        "status": "SUCCESS",
        "pre_aligned": pre_aligned,
        "post_aligned": post_aligned,
        "overlap_mask": overlap_mask,
        "metadata": alignment_metadata,
        "warnings": comp["warnings"],
    }


def generate_alignment_diagnostics(
    pre_aligned: np.ndarray,
    post_aligned: np.ndarray,
    output_dir: Union[str, Path],
    prefix: str = "SpaceNet8_Germany_0_16_67",
    square_size: int = 130
) -> Dict[str, Path]:
    """
    Generates and saves visual alignment diagnostic products:
    1. Side-by-side comparison
    2. 50/50 alpha overlay
    3. Checkerboard composite
    4. Pixel-wise absolute difference

    Parameters:
        pre_aligned: Aligned PRE image array (H, W, 3) or (H, W).
        post_aligned: Aligned POST image array (H, W, 3) or (H, W).
        output_dir: Directory where diagnostic images will be stored.
        prefix: Naming prefix for generated files.
        square_size: Size in pixels of checkerboard grid squares.

    Returns:
        Dict mapping diagnostic names to saved file Paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Ensure 3-channel uint8 for visualization
    def to_vis_rgb(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            # Normalize float or int to uint8
            finite_m = np.isfinite(img)
            min_v = np.min(img[finite_m]) if np.any(finite_m) else 0.0
            max_v = np.max(img[finite_m]) if np.any(finite_m) else 1.0
            diff = max_v - min_v if max_v > min_v else 1.0
            norm = np.clip((img - min_v) / diff * 255.0, 0, 255).astype(np.uint8)
        else:
            norm = img

        if norm.ndim == 2:
            return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        elif norm.ndim == 3 and norm.shape[2] == 1:
            return cv2.cvtColor(norm[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif norm.ndim == 3 and norm.shape[2] >= 3:
            return norm[:, :, :3]
        return norm

    pre_vis = to_vis_rgb(pre_aligned)
    post_vis = to_vis_rgb(post_aligned)

    h, w = pre_vis.shape[:2]

    # 1. Side-by-side
    side_by_side = np.hstack([pre_vis, post_vis])
    p_side = out_path / f"{prefix}_alignment_side_by_side.png"
    cv2.imwrite(str(p_side), side_by_side)

    # 2. Alpha overlay
    overlay = cv2.addWeighted(pre_vis, 0.5, post_vis, 0.5, 0.0)
    p_overlay = out_path / f"{prefix}_alignment_overlay.png"
    cv2.imwrite(str(p_overlay), overlay)

    # 3. Checkerboard
    checkerboard = pre_vis.copy()
    for y in range(0, h, square_size):
        for x in range(0, w, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 1:
                checkerboard[y:min(y + square_size, h), x:min(x + square_size, w)] = \
                    post_vis[y:min(y + square_size, h), x:min(x + square_size, w)]
    p_checker = out_path / f"{prefix}_alignment_checkerboard.png"
    cv2.imwrite(str(p_checker), checkerboard)

    # 4. Absolute difference
    absdiff = cv2.absdiff(pre_vis, post_vis)
    p_diff = out_path / f"{prefix}_alignment_absdiff.png"
    cv2.imwrite(str(p_diff), absdiff)

    return {
        "side_by_side": p_side,
        "overlay": p_overlay,
        "checkerboard": p_checker,
        "absdiff": p_diff,
    }


def save_alignment_report(
    metadata: Dict[str, Any],
    output_path: Union[str, Path]
) -> Path:
    """Saves alignment metadata to a clean JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path
