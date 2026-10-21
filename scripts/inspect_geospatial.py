"""
FloodScope AI - Geospatial Inspection Tool
Member 1: Dataset Preparation and Verification

Uses rasterio to deeply inspect dimensions, Coordinate Reference Systems (CRS),
geotransforms, pixel resolution, bounding boxes, and band counts of the SpaceNet 8
pre-event and post-event GeoTIFFs, as well as parsing the GeoJSON annotations.
"""

import json
from pathlib import Path
import rasterio

PRE_PATH = Path("dataset/raw/SpaceNet8_Germany_0_16_67_PRE.tif")
POST_PATH = Path("dataset/raw/SpaceNet8_Germany_0_16_67_POST.tif")
GEOJSON_PATH = Path("dataset/labels/SpaceNet8_Germany_0_16_67_annotations.geojson")


def inspect_raster(path: Path):
    with rasterio.open(path) as src:
        meta = {
            "file": path.name,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtypes": src.dtypes,
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "res": src.res,
            "nodata": src.nodata,
        }
    return meta


def inspect_geojson(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    geom_types = {}
    flooded_counts = {"flooded_true": 0, "flooded_false": 0, "null_or_unspecified": 0}
    building_count = 0
    highway_count = 0

    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type", "Unknown")
        geom_types[gtype] = geom_types.get(gtype, 0) + 1

        props = feat.get("properties", {})
        if props.get("flooded") is True or props.get("flooded") == "flooded":
            flooded_counts["flooded_true"] += 1
        elif props.get("flooded") is False or props.get("flooded") == "non-flooded":
            flooded_counts["flooded_false"] += 1
        else:
            flooded_counts["null_or_unspecified"] += 1

        if props.get("building"):
            building_count += 1
        if props.get("highway"):
            highway_count += 1

    crs_info = data.get("crs", "OGC:CRS84 (Default WGS84 Geographic Lat/Lon in GeoJSON spec)")

    return {
        "file": path.name,
        "total_features": len(features),
        "geometry_types": geom_types,
        "building_features": building_count,
        "highway_features": highway_count,
        "flooded_counts": flooded_counts,
        "crs": crs_info,
    }


def main():
    print("=" * 70)
    print("SpaceNet 8 (0_16_67) Geospatial & Compatibility Inspection")
    print("=" * 70)

    pre_meta = inspect_raster(PRE_PATH)
    post_meta = inspect_raster(POST_PATH)

    print("\n--- Pre-Event GeoTIFF ---")
    for k, v in pre_meta.items():
        print(f"  {k}: {v}")

    print("\n--- Post-Event GeoTIFF ---")
    for k, v in post_meta.items():
        print(f"  {k}: {v}")

    print("\n--- Spatial Compatibility Comparison ---")
    compat_dims = (pre_meta["width"] == post_meta["width"]) and (pre_meta["height"] == post_meta["height"])
    compat_crs = pre_meta["crs"] == post_meta["crs"]
    compat_bounds = pre_meta["bounds"] == post_meta["bounds"]
    compat_transform = pre_meta["transform"] == post_meta["transform"]

    print(f"  Dimensions Match: {compat_dims} ({pre_meta['width']}x{pre_meta['height']} vs {post_meta['width']}x{post_meta['height']})")
    print(f"  CRS Match:        {compat_crs} ({pre_meta['crs']} vs {post_meta['crs']})")
    print(f"  Transform Match:  {compat_transform}")
    print(f"  Bounds Match:     {compat_bounds}")

    print("\n--- GeoJSON Annotations Inspection ---")
    geojson_info = inspect_geojson(GEOJSON_PATH)
    for k, v in geojson_info.items():
        print(f"  {k}: {v}")

    print("=" * 70)


if __name__ == "__main__":
    main()
