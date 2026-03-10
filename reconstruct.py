#!/usr/bin/env python3
"""
COLMAP Berry Reconstruction Pipeline

Takes a capture session directory (from Rowfollow.py --capture or colmap_capture.py),
runs COLMAP sparse reconstruction, detects raspberries via HSV color segmentation,
maps 2D detections to 3D coordinates, and exports berries.json for the farm twin UI.

Usage:
    python reconstruct.py colmap_sessions/session_20260214_153012

Requires COLMAP installed (brew install colmap / apt install colmap).
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# Stage 1: COLMAP sparse reconstruction
# =============================================================================

def run_colmap(session_dir: Path) -> Path:
    """
    Run COLMAP feature extraction, matching, and sparse reconstruction.
    Returns the path to the reconstruction output directory.
    """
    image_path = session_dir / "images"
    db_path = session_dir / "database.db"
    sparse_path = session_dir / "sparse"
    sparse_path.mkdir(exist_ok=True)

    if not image_path.is_dir() or not any(image_path.iterdir()):
        print(f"[ERROR] No images found in {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("Stage 1: COLMAP Sparse Reconstruction")
    print("=" * 60)

    # Feature extraction
    print("\n[COLMAP] Feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_path),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", "1",
    ], check=True)

    # Exhaustive matching (fine for < ~500 images typical of a row pass)
    print("\n[COLMAP] Exhaustive matching...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
    ], check=True)

    # Sparse mapper
    print("\n[COLMAP] Sparse mapper...")
    subprocess.run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_path),
        "--output_path", str(sparse_path),
    ], check=True)

    # COLMAP creates numbered subdirectories (0, 1, ...) for each model.
    # Pick the largest one (most images registered).
    model_dirs = sorted(sparse_path.iterdir())
    if not model_dirs:
        print("[ERROR] COLMAP produced no reconstruction.")
        sys.exit(1)

    best_model = model_dirs[0]
    print(f"\n[COLMAP] Using model: {best_model}")

    # Export as text if binary
    txt_marker = best_model / "cameras.txt"
    if not txt_marker.exists():
        print("[COLMAP] Converting binary model to text...")
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", str(best_model),
            "--output_path", str(best_model),
            "--output_type", "TXT",
        ], check=True)

    return best_model


# =============================================================================
# COLMAP text-file parsers
# =============================================================================

def parse_cameras(path: Path) -> Dict[int, Dict[str, Any]]:
    """Parse cameras.txt → {camera_id: {model, width, height, params}}"""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            cameras[cam_id] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(x) for x in parts[4:]],
            }
    return cameras


def parse_images(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse images.txt → {image_name: {image_id, camera_id, qvec, tvec, keypoints}}
    keypoints is a list of (x, y, point3d_id) tuples.
    """
    images = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # images.txt has pairs of lines: header then keypoints
    for i in range(0, len(lines), 2):
        header = lines[i].split()
        img_id = int(header[0])
        qw, qx, qy, qz = float(header[1]), float(header[2]), float(header[3]), float(header[4])
        tx, ty, tz = float(header[5]), float(header[6]), float(header[7])
        cam_id = int(header[8])
        name = header[9]

        kp_line = lines[i + 1].split() if i + 1 < len(lines) else []
        keypoints = []
        for j in range(0, len(kp_line), 3):
            x = float(kp_line[j])
            y = float(kp_line[j + 1])
            p3d_id = int(kp_line[j + 2])
            keypoints.append((x, y, p3d_id))

        images[name] = {
            "image_id": img_id,
            "camera_id": cam_id,
            "qvec": (qw, qx, qy, qz),
            "tvec": (tx, ty, tz),
            "keypoints": keypoints,
        }

    return images


def parse_points3d(path: Path) -> Dict[int, Dict[str, Any]]:
    """Parse points3D.txt → {point3d_id: {xyz, rgb, error}}"""
    points = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            p_id = int(parts[0])
            xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
            rgb = (int(parts[4]), int(parts[5]), int(parts[6]))
            error = float(parts[7])
            points[p_id] = {"xyz": xyz, "rgb": rgb, "error": error}
    return points


# =============================================================================
# Stage 2: Raspberry detection via HSV color segmentation
# =============================================================================

# HSV ranges for raspberries (OpenCV H is 0-179)
RASPBERRY_RANGES = {
    "ripe": [
        # Deep red / crimson (hue wraps around 0)
        ((0, 80, 60), (10, 255, 255)),
        ((160, 80, 60), (179, 255, 255)),
    ],
    "ripening": [
        # Pink / light red / orange tones
        ((5, 40, 100), (25, 200, 255)),
    ],
    "unripe": [
        # Green
        ((35, 40, 40), (80, 255, 200)),
    ],
}

MIN_BLOB_AREA = 50        # px²  — ignore tiny noise
MAX_BLOB_AREA = 50000     # px²  — ignore huge false positives


def detect_berries_in_image(
    img_bgr: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Detect raspberry blobs via HSV thresholding.
    Returns list of {cx, cy, area, status, bbox} dicts.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    detections = []

    for status, ranges in RASPBERRY_RANGES.items():
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8
        )

        for lbl in range(1, n_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
                continue
            cx, cy = centroids[lbl]
            bx = stats[lbl, cv2.CC_STAT_LEFT]
            by = stats[lbl, cv2.CC_STAT_TOP]
            bw = stats[lbl, cv2.CC_STAT_WIDTH]
            bh = stats[lbl, cv2.CC_STAT_HEIGHT]
            detections.append({
                "cx": cx, "cy": cy, "area": area, "status": status,
                "bbox": (bx, by, bw, bh),
            })

    return detections


def detect_all_images(image_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Run berry detection on every image. Returns {filename: [detections]}."""
    print("\n" + "=" * 60)
    print("Stage 2: Berry Detection (HSV)")
    print("=" * 60)

    results = {}
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    total = len(image_files)

    for idx, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets = detect_berries_in_image(img)
        results[img_path.name] = dets
        if (idx + 1) % 10 == 0 or idx == total - 1:
            det_count = sum(len(v) for v in results.values())
            print(f"  [{idx+1}/{total}] processed — {det_count} detections so far")

    total_dets = sum(len(v) for v in results.values())
    print(f"\n  Total detections across {total} images: {total_dets}")
    return results


# =============================================================================
# Stage 3: Map 2D detections → 3D via COLMAP keypoints
# =============================================================================

MATCH_RADIUS = 15.0  # max pixel distance from blob centroid to COLMAP keypoint


def map_detections_to_3d(
    detections: Dict[str, List[Dict[str, Any]]],
    colmap_images: Dict[str, Dict[str, Any]],
    points3d: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each detected berry blob, find the nearest COLMAP 2D keypoint
    that has a valid 3D point, then collect the 3D coordinate.
    Deduplicates by POINT3D_ID across images.
    """
    print("\n" + "=" * 60)
    print("Stage 3: 2D → 3D Mapping")
    print("=" * 60)

    # point3d_id → best detection info (keep highest confidence = smallest distance)
    point_map: Dict[int, Dict[str, Any]] = {}

    matched = 0
    unmatched = 0

    for img_name, dets in detections.items():
        if img_name not in colmap_images:
            continue
        img_data = colmap_images[img_name]
        kps = img_data["keypoints"]

        # Build arrays for fast nearest-neighbour
        valid_kps = [(x, y, pid) for x, y, pid in kps if pid != -1 and pid in points3d]
        if not valid_kps:
            continue
        kp_xy = np.array([(x, y) for x, y, _ in valid_kps], dtype=np.float32)
        kp_ids = [pid for _, _, pid in valid_kps]

        for det in dets:
            cx, cy = det["cx"], det["cy"]
            dists = np.sqrt((kp_xy[:, 0] - cx) ** 2 + (kp_xy[:, 1] - cy) ** 2)
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]

            if best_dist > MATCH_RADIUS:
                # Also try within bounding box
                bx, by, bw, bh = det["bbox"]
                in_bbox = (
                    (kp_xy[:, 0] >= bx) & (kp_xy[:, 0] <= bx + bw) &
                    (kp_xy[:, 1] >= by) & (kp_xy[:, 1] <= by + bh)
                )
                if not np.any(in_bbox):
                    unmatched += 1
                    continue
                bbox_dists = np.where(in_bbox, dists, 1e9)
                best_idx = int(np.argmin(bbox_dists))
                best_dist = bbox_dists[best_idx]

            pid = kp_ids[best_idx]
            # Keep the detection with smallest distance to keypoint
            if pid not in point_map or best_dist < point_map[pid]["_dist"]:
                pt3d = points3d[pid]
                point_map[pid] = {
                    "xyz": pt3d["xyz"],
                    "status": det["status"],
                    "area_px": det["area"],
                    "_dist": best_dist,
                }
            matched += 1

    print(f"  Matched detections: {matched}")
    print(f"  Unmatched detections: {unmatched}")
    print(f"  Unique 3D berry points: {len(point_map)}")

    # Build output list
    berries = []
    for idx, (pid, info) in enumerate(sorted(point_map.items()), start=1):
        x, y, z = info["xyz"]
        # Rough size estimate from blob area (assuming ~2mm/px at typical distance)
        size_mm = max(5, min(80, int(np.sqrt(info["area_px"]) * 1.5)))
        # Confidence from keypoint match distance (closer = better)
        conf = max(0.5, min(1.0, 1.0 - info["_dist"] / (MATCH_RADIUS * 2)))
        berries.append({
            "id": idx,
            "x": round(x, 4),
            "y": round(y, 4),
            "z": round(z, 4),
            "status": info["status"],
            "confidence": round(conf, 2),
            "size_mm": size_mm,
        })

    return berries


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="COLMAP berry reconstruction pipeline"
    )
    parser.add_argument(
        "session", type=str,
        help="Path to capture session directory (e.g. colmap_sessions/session_...)"
    )
    parser.add_argument(
        "--skip-colmap", action="store_true",
        help="Skip COLMAP reconstruction (use existing sparse/ output)"
    )
    parser.add_argument(
        "--match-radius", type=float, default=MATCH_RADIUS,
        help="Max pixel distance for matching detections to keypoints"
    )
    parser.add_argument(
        "--min-blob", type=int, default=MIN_BLOB_AREA,
        help="Minimum blob area in pixels"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: <session>/berries.json)"
    )
    args = parser.parse_args()

    global MATCH_RADIUS, MIN_BLOB_AREA
    MATCH_RADIUS = args.match_radius
    MIN_BLOB_AREA = args.min_blob

    session_dir = Path(args.session)
    if not session_dir.is_dir():
        print(f"[ERROR] Session directory not found: {session_dir}")
        sys.exit(1)

    image_dir = session_dir / "images"
    if not image_dir.is_dir():
        print(f"[ERROR] No images/ folder in {session_dir}")
        sys.exit(1)

    # --- Stage 1: COLMAP ---
    if args.skip_colmap:
        sparse_path = session_dir / "sparse"
        model_dirs = sorted(sparse_path.iterdir()) if sparse_path.is_dir() else []
        if not model_dirs:
            print("[ERROR] --skip-colmap but no sparse/ model found")
            sys.exit(1)
        model_path = model_dirs[0]
        print(f"[INFO] Skipping COLMAP, using existing model: {model_path}")
    else:
        model_path = run_colmap(session_dir)

    # --- Parse COLMAP outputs ---
    print("\n[PARSE] Reading COLMAP text files...")
    cameras = parse_cameras(model_path / "cameras.txt")
    colmap_images = parse_images(model_path / "images.txt")
    points3d = parse_points3d(model_path / "points3D.txt")
    print(f"  Cameras: {len(cameras)}")
    print(f"  Registered images: {len(colmap_images)}")
    print(f"  3D points: {len(points3d)}")

    # --- Stage 2: Berry detection ---
    detections = detect_all_images(image_dir)

    # --- Stage 3: 2D → 3D mapping ---
    berries = map_detections_to_3d(detections, colmap_images, points3d)

    # --- Export ---
    out_path = Path(args.output) if args.output else session_dir / "berries.json"
    with open(out_path, "w") as f:
        json.dump(berries, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Done! {len(berries)} berries exported to {out_path}")
    print("=" * 60)

    counts = defaultdict(int)
    for b in berries:
        counts[b["status"]] += 1
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")

    print(f"\nOpen farm_twin.html and load {out_path.name} to visualize.")


if __name__ == "__main__":
    main()
