"""
ORB-SLAM2 from Scratch — Step 1: Processing the First Frame
============================================================

This script walks through exactly what ORB-SLAM2 does when the monocular
camera delivers its very first frame. We implement each stage by hand
(where instructive) and with OpenCV (where practical).

Dataset assumption:
    frames/0.png, frames/1.png, frames/2.png, ...

Usage:
    python orbslam2_step1_first_frame.py --frames_dir ./frames
"""

import cv2
import numpy as np
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
#    Before processing anything, we define the containers that
#    ORB-SLAM2 uses to store what it learns about the world.
# ─────────────────────────────────────────────────────────────

@dataclass
class KeyPoint:
    """One detected feature in an image.

    This is what ORB-SLAM2 extracts from every frame — a corner in the
    image with enough structure around it to be recognized later.
    """
    x: float                  # pixel column
    y: float                  # pixel row
    octave: int               # which pyramid level it was found at (0 = full res)
    angle: float              # orientation in degrees (from intensity centroid)
    response: float           # Harris corner score (higher = more "corner-like")
    descriptor: np.ndarray    # 256-bit binary string (32 bytes) — the fingerprint


@dataclass
class KeyFrame:
    """A frame the system decides is worth keeping.

    Not every frame becomes a keyframe. But frame 0 always does — it's the
    reference that defines the world coordinate origin.
    """
    id: int
    image: np.ndarray                    # grayscale image
    keypoints: List[KeyPoint]            # all detected ORB features
    pose: np.ndarray                     # 4×4 camera-to-world transform
    pyramid: List[np.ndarray] = None     # image pyramid levels


@dataclass
class Map:
    """The 3D map that ORB-SLAM2 builds incrementally.

    After frame 0, this is empty — we need two views to triangulate.
    """
    keyframes: List[KeyFrame] = field(default_factory=list)
    map_points: List[np.ndarray] = field(default_factory=list)  # Nx3 array of 3D points


# ─────────────────────────────────────────────────────────────
# 2. IMAGE PYRAMID
#    ORB-SLAM2 builds an 8-level Gaussian pyramid so that it
#    can detect features at multiple scales. A distant object
#    might only be recognizable at a coarser pyramid level.
# ─────────────────────────────────────────────────────────────

def build_image_pyramid(
    image: np.ndarray,
    n_levels: int = 8,
    scale_factor: float = 1.2,
) -> List[np.ndarray]:
    """Build a scale-space pyramid by repeatedly downscaling.

    Args:
        image:        Grayscale input image (H, W).
        n_levels:     Number of pyramid levels (ORB-SLAM2 default: 8).
        scale_factor: Scale ratio between consecutive levels (default: 1.2).

    Returns:
        List of images, from finest (level 0 = original) to coarsest.

    Why 1.2 and not 2.0?
        A factor of 2 (like SIFT) is too coarse — you'd miss features that
        fall between scales. 1.2 gives denser scale sampling, which means
        more reliable matching across viewpoints with different distances.
    """
    pyramid = [image]
    for level in range(1, n_levels):
        # Each level is 1/scale_factor the size of the previous one
        inv_scale = 1.0 / (scale_factor ** level)
        width = int(image.shape[1] * inv_scale)
        height = int(image.shape[0] * inv_scale)
        scaled = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        pyramid.append(scaled)

    return pyramid


def print_pyramid_info(pyramid: List[np.ndarray]):
    """Display the pyramid structure — lets you see the scale cascade."""
    print("\n📐 Image pyramid:")
    print(f"   {'Level':<8} {'Size':<16} {'Scale':<10} {'Pixels':>10}")
    print("   " + "─" * 48)
    base_pixels = pyramid[0].shape[0] * pyramid[0].shape[1]
    for i, level in enumerate(pyramid):
        h, w = level.shape[:2]
        scale = (h * w) / base_pixels
        print(f"   {i:<8} {w}×{h:<11} {scale:.3f}     {h * w:>10,}")


# ─────────────────────────────────────────────────────────────
# 3. FAST CORNER DETECTION (manual implementation)
#    This is the raw detector — check 16 pixels on a circle
#    around each candidate. If ≥9 consecutive are brighter
#    (or darker) than the center, it's a corner.
#
#    We implement a simplified version here for understanding,
#    then use OpenCV's optimized version for the real pipeline.
# ─────────────────────────────────────────────────────────────

# The 16 pixel offsets on the Bresenham circle of radius 3
FAST_CIRCLE = [
    (0, -3), (1, -3), (2, -2), (3, -1),   # positions 1-4
    (3, 0),  (3, 1),  (2, 2),  (1, 3),     # positions 5-8
    (0, 3),  (-1, 3), (-2, 2), (-3, 1),    # positions 9-12
    (-3, 0), (-3, -1),(-2, -2),(-1, -3),   # positions 13-16
]


def fast_corner_test(image: np.ndarray, x: int, y: int, threshold: int = 20) -> bool:
    """Test if pixel (x, y) is a FAST-9 corner.

    This is the conceptual implementation — in practice OpenCV uses
    a decision tree that checks pixels 1, 5, 9, 13 first (the compass
    points) to quickly reject non-corners, making it ~10x faster.

    Args:
        image:     Grayscale image.
        x, y:      Pixel coordinates to test.
        threshold: Intensity difference threshold.

    Returns:
        True if ≥9 consecutive circle pixels are all brighter or all darker.
    """
    center = int(image[y, x])
    bright_threshold = center + threshold
    dark_threshold = center - threshold

    # Classify each of the 16 circle pixels
    # +1 = brighter, -1 = darker, 0 = similar
    states = []
    for dx, dy in FAST_CIRCLE:
        px, py = x + dx, y + dy
        if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
            val = int(image[py, px])
            if val > bright_threshold:
                states.append(1)
            elif val < dark_threshold:
                states.append(-1)
            else:
                states.append(0)
        else:
            states.append(0)

    # Check for ≥9 consecutive bright OR ≥9 consecutive dark
    # We wrap around: check in a doubled array
    doubled = states + states
    for target in [1, -1]:
        run = 0
        for s in doubled:
            if s == target:
                run += 1
                if run >= 9:
                    return True
            else:
                run = 0

    return False


def detect_fast_manual(image: np.ndarray, threshold: int = 20, max_corners: int = 200):
    """Run FAST on every pixel (slow but educational).

    In practice, ORB-SLAM2 uses OpenCV's optimized FAST with non-maximum
    suppression. This manual version helps you understand the core idea.
    """
    h, w = image.shape
    corners = []
    margin = 3  # radius of the Bresenham circle

    for y in range(margin, h - margin):
        for x in range(margin, w - margin):
            if fast_corner_test(image, x, y, threshold):
                corners.append((x, y))

    print(f"   Manual FAST found {len(corners)} corners (before filtering)")
    return corners[:max_corners]


# ─────────────────────────────────────────────────────────────
# 4. GRID-BASED KEYPOINT DISTRIBUTION
#    ORB-SLAM2 doesn't just take the top N keypoints globally —
#    that would cluster them all on textured objects and leave
#    blank walls with zero coverage. Instead it divides the
#    image into a grid and takes the best from each cell.
# ─────────────────────────────────────────────────────────────

def distribute_keypoints_grid(
    keypoints: list,
    image_shape: tuple,
    n_rows: int = 6,
    n_cols: int = 8,
    max_per_cell: int = 30,
) -> list:
    """Select keypoints ensuring spatial coverage across the image.

    Args:
        keypoints:    List of cv2.KeyPoint objects.
        image_shape:  (H, W) of the image.
        n_rows:       Number of grid rows.
        n_cols:       Number of grid columns.
        max_per_cell: Maximum keypoints to keep per cell.

    Returns:
        Filtered list of keypoints with even spatial distribution.
    """
    h, w = image_shape[:2]
    cell_h = h / n_rows
    cell_w = w / n_cols

    # Bin keypoints into grid cells
    grid = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    for kp in keypoints:
        col = min(int(kp.pt[0] / cell_w), n_cols - 1)
        row = min(int(kp.pt[1] / cell_h), n_rows - 1)
        grid[row][col].append(kp)

    # From each cell, take the top `max_per_cell` by response (Harris score)
    distributed = []
    occupied_cells = 0
    for row in range(n_rows):
        for col in range(n_cols):
            cell = grid[row][col]
            if cell:
                occupied_cells += 1
                cell.sort(key=lambda kp: kp.response, reverse=True)
                distributed.extend(cell[:max_per_cell])

    total_cells = n_rows * n_cols
    print(f"   Grid distribution: {occupied_cells}/{total_cells} cells occupied")
    print(f"   Keypoints after filtering: {len(distributed)}")

    return distributed


# ─────────────────────────────────────────────────────────────
# 5. ORIENTATION (intensity centroid)
#    OpenCV's ORB does this internally, but let's implement it
#    so you can see what atan2(m01, m10) actually means.
# ─────────────────────────────────────────────────────────────

def compute_orientation(image: np.ndarray, x: int, y: int, patch_radius: int = 15) -> float:
    """Compute keypoint orientation via the intensity centroid method.

    The idea: treat pixel intensities as "mass". Find the center of mass.
    The angle from the geometric center to this center of mass is a
    repeatable orientation — it rotates with the image.

    Args:
        image:        Grayscale image.
        x, y:         Keypoint position.
        patch_radius: Radius of the circular patch to analyze.

    Returns:
        Orientation angle in degrees.
    """
    m10 = 0.0  # moment: sum of (intensity * dx)
    m01 = 0.0  # moment: sum of (intensity * dy)

    h, w = image.shape
    for dy in range(-patch_radius, patch_radius + 1):
        for dx in range(-patch_radius, patch_radius + 1):
            # Only use pixels inside the circular patch
            if dx * dx + dy * dy > patch_radius * patch_radius:
                continue
            px, py = x + dx, y + dy
            if 0 <= px < w and 0 <= py < h:
                intensity = float(image[py, px])
                m10 += intensity * dx
                m01 += intensity * dy

    angle = np.degrees(np.arctan2(m01, m10))
    return angle


# ─────────────────────────────────────────────────────────────
# 6. FULL ORB EXTRACTION (using OpenCV)
#    For the actual pipeline we use OpenCV's ORB, which does
#    all of the above (pyramid, FAST, orientation, rBRIEF)
#    in a single optimized call. We wrap it with our grid
#    distribution logic.
# ─────────────────────────────────────────────────────────────

def extract_orb_features(
    image: np.ndarray,
    n_features: int = 2000,
    n_levels: int = 8,
    scale_factor: float = 1.2,
) -> tuple:
    """Extract ORB features with grid-based distribution.

    This is what ORB-SLAM2's tracking thread calls on every frame.

    Args:
        image:        Grayscale input image.
        n_features:   Target number of features.
        n_levels:     Pyramid levels.
        scale_factor: Pyramid scale factor.

    Returns:
        (keypoints, descriptors) — keypoints are cv2.KeyPoint objects,
        descriptors is an (N, 32) uint8 array (256 bits per row).
    """
    # Create the ORB detector
    # - WTA_K=2 means each BRIEF test compares 2 pixels (standard)
    # - scoreType=HARRIS uses Harris corner score for ranking
    # - patchSize=31 is the patch used for descriptor computation
    orb = cv2.ORB_create(
        nfeatures=n_features * 2,   # detect more, then filter via grid
        scaleFactor=scale_factor,
        nlevels=n_levels,
        edgeThreshold=19,           # border margin where no features are detected
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    )

    # Detect keypoints and compute descriptors in one call
    # Internally this: builds pyramid → FAST at each level →
    # orientation via centroid → rotated BRIEF descriptor
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if keypoints is None or len(keypoints) == 0:
        print("   ⚠ No keypoints detected!")
        return [], None

    print(f"   Raw ORB detection: {len(keypoints)} keypoints")

    # Apply grid-based distribution for even spatial coverage
    keypoints = distribute_keypoints_grid(keypoints, image.shape)

    # Recompute descriptors for the filtered set
    # (we need to pass the keypoints back through ORB to get aligned descriptors)
    keypoints, descriptors = orb.compute(image, keypoints)

    return keypoints, descriptors


# ─────────────────────────────────────────────────────────────
# 7. DESCRIPTOR ANALYSIS
#    Let's peek inside the binary descriptors to build intuition
#    for how matching will work in later frames.
# ─────────────────────────────────────────────────────────────

def analyze_descriptors(descriptors: np.ndarray, n_samples: int = 5):
    """Inspect the binary descriptors — what do they actually look like?"""
    print(f"\n🔍 Descriptor analysis:")
    print(f"   Shape: {descriptors.shape}  →  {descriptors.shape[0]} keypoints × {descriptors.shape[1]} bytes")
    print(f"   Bits per descriptor: {descriptors.shape[1] * 8}")

    print(f"\n   First {n_samples} descriptors (as binary):")
    for i in range(min(n_samples, len(descriptors))):
        # Convert each byte to 8 bits
        bits = ''.join(f'{byte:08b}' for byte in descriptors[i])
        # Show first 64 bits (out of 256) for readability
        print(f"   [{i:4d}] {bits[:64]}... ({bits.count('1')}/256 bits set)")

    # Compute pairwise Hamming distances for a sample
    print(f"\n   Hamming distances between first {n_samples} descriptors:")
    print(f"   (this is how matching works — lower = more similar)")
    print(f"   {'':>8}", end="")
    for j in range(min(n_samples, len(descriptors))):
        print(f"  [{j}]", end="")
    print()
    for i in range(min(n_samples, len(descriptors))):
        print(f"   [{i}]   ", end="")
        for j in range(min(n_samples, len(descriptors))):
            # Hamming distance = number of differing bits
            dist = np.sum(np.unpackbits(np.bitwise_xor(descriptors[i], descriptors[j])))
            print(f"  {dist:3d}", end="")
        print()


# ─────────────────────────────────────────────────────────────
# 8. CONVERT TO BoW (Bag of Words)
#    ORB-SLAM2 also converts every keyframe's descriptors into
#    a bag-of-words vector using DBoW2. This is used later for
#    loop closure detection and fast keyframe matching.
#
#    We simulate the concept here — a full implementation would
#    require a pre-trained vocabulary tree (typically trained
#    on a large dataset of ORB descriptors).
# ─────────────────────────────────────────────────────────────

def compute_bow_placeholder(descriptors: np.ndarray) -> dict:
    """Placeholder for the DBoW2 bag-of-words conversion.

    In ORB-SLAM2, this uses a pre-trained vocabulary tree with ~1M visual
    words organized in 6 levels with branching factor 10. Each descriptor
    is quantized to the nearest visual word, producing a sparse histogram.

    For now we cluster the descriptors to illustrate the concept.
    """
    if descriptors is None or len(descriptors) < 10:
        return {}

    # K-means on binary descriptors (using Hamming-friendly approach)
    # In real DBoW2, this is a hierarchical vocabulary tree, not flat k-means
    n_words = min(50, len(descriptors) // 4)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    descriptors_float = descriptors.astype(np.float32)
    _, labels, centers = cv2.kmeans(
        descriptors_float, n_words, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    # Build the histogram (bag of words)
    bow_vector = np.zeros(n_words)
    for label in labels.flatten():
        bow_vector[label] += 1

    # Normalize (TF-IDF style — here just L1 for simplicity)
    bow_vector /= bow_vector.sum()

    print(f"\n📚 Bag of words (simplified):")
    print(f"   Vocabulary size: {n_words} visual words")
    print(f"   Non-zero entries: {np.count_nonzero(bow_vector)}/{n_words}")
    print(f"   Top 5 words: {np.argsort(bow_vector)[-5:][::-1]}")

    return {"vector": bow_vector, "labels": labels, "centers": centers}


# ─────────────────────────────────────────────────────────────
# 9. BUILD THE REFERENCE KEYFRAME
#    Everything comes together here. We package the extracted
#    features into a KeyFrame object and set the pose to identity
#    (this frame defines the world origin).
# ─────────────────────────────────────────────────────────────

def create_reference_keyframe(
    image: np.ndarray,
    keypoints_cv: list,
    descriptors: np.ndarray,
    pyramid: List[np.ndarray],
) -> KeyFrame:
    """Create keyframe 0 — the origin of the world.

    The pose is set to the 4×4 identity matrix, meaning:
    - The camera is at position (0, 0, 0)
    - It's looking along the +Z axis
    - The world coordinate system is defined by this first view

    Everything ORB-SLAM2 builds from here on is relative to this frame.
    """
    # Convert OpenCV keypoints to our dataclass
    kps = []
    for i, kp in enumerate(keypoints_cv):
        kps.append(KeyPoint(
            x=kp.pt[0],
            y=kp.pt[1],
            octave=kp.octave,
            angle=kp.angle,
            response=kp.response,
            descriptor=descriptors[i] if descriptors is not None else None,
        ))

    # Identity pose: this camera IS the world origin
    pose = np.eye(4, dtype=np.float64)

    keyframe = KeyFrame(
        id=0,
        image=image,
        keypoints=kps,
        pose=pose,
        pyramid=pyramid,
    )

    return keyframe


# ─────────────────────────────────────────────────────────────
# 10. VISUALIZATION
#     Draw what we've extracted so you can see it.
# ─────────────────────────────────────────────────────────────

def visualize_results(
    image: np.ndarray,
    keypoints_cv: list,
    pyramid: List[np.ndarray],
    output_dir: str = "./output",
):
    """Save visualizations of each processing stage."""
    os.makedirs(output_dir, exist_ok=True)

    # (a) Keypoints on the original image
    img_kp = cv2.drawKeypoints(
        image, keypoints_cv, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(os.path.join(output_dir, "01_keypoints.png"), img_kp)

    # (b) Image pyramid — stitch levels side by side
    max_h = pyramid[0].shape[0]
    total_w = sum(level.shape[1] for level in pyramid) + (len(pyramid) - 1) * 5
    canvas = np.zeros((max_h, total_w), dtype=np.uint8)
    x_offset = 0
    for level in pyramid:
        h, w = level.shape
        canvas[:h, x_offset:x_offset + w] = level
        x_offset += w + 5
    cv2.imwrite(os.path.join(output_dir, "02_pyramid.png"), canvas)

    # (c) Keypoints colored by pyramid level (octave)
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    octave_colors = [
        (255, 50, 50),    # level 0: red
        (50, 255, 50),    # level 1: green
        (50, 50, 255),    # level 2: blue
        (255, 255, 50),   # level 3: yellow
        (255, 50, 255),   # level 4: magenta
        (50, 255, 255),   # level 5: cyan
        (200, 150, 50),   # level 6: orange
        (150, 50, 200),   # level 7: purple
    ]
    for kp in keypoints_cv:
        color = octave_colors[min(kp.octave, len(octave_colors) - 1)]
        center = (int(kp.pt[0]), int(kp.pt[1]))
        radius = int(kp.size / 2)
        cv2.circle(img_color, center, max(radius, 2), color, 1)
        # Draw orientation line
        angle_rad = np.radians(kp.angle)
        end = (
            int(center[0] + radius * np.cos(angle_rad)),
            int(center[1] + radius * np.sin(angle_rad)),
        )
        cv2.line(img_color, center, end, color, 1)
    cv2.imwrite(os.path.join(output_dir, "03_keypoints_by_octave.png"), img_color)

    # (d) Grid coverage heatmap
    h, w = image.shape[:2]
    n_rows, n_cols = 6, 8
    cell_h, cell_w = h / n_rows, w / n_cols
    grid_counts = np.zeros((n_rows, n_cols), dtype=int)
    for kp in keypoints_cv:
        col = min(int(kp.pt[0] / cell_w), n_cols - 1)
        row = min(int(kp.pt[1] / cell_h), n_rows - 1)
        grid_counts[row, col] += 1

    # Upscale heatmap to image size
    heatmap = grid_counts.astype(np.float32)
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)

    # Overlay grid lines
    for r in range(1, n_rows):
        y = int(r * cell_h)
        cv2.line(heatmap_color, (0, y), (w, y), (255, 255, 255), 1)
    for c in range(1, n_cols):
        x = int(c * cell_w)
        cv2.line(heatmap_color, (x, 0), (x, h), (255, 255, 255), 1)

    cv2.imwrite(os.path.join(output_dir, "04_grid_coverage.png"), heatmap_color)

    print(f"\n📸 Visualizations saved to {output_dir}/")
    print(f"   01_keypoints.png         — all ORB features with size and orientation")
    print(f"   02_pyramid.png           — the 8-level image pyramid")
    print(f"   03_keypoints_by_octave.png — features colored by pyramid level")
    print(f"   04_grid_coverage.png     — heatmap of spatial distribution")


# ─────────────────────────────────────────────────────────────
# 11. MAIN — ORCHESTRATE EVERYTHING
# ─────────────────────────────────────────────────────────────

def process_first_frame(image_path: str) -> tuple:
    """Run the complete ORB-SLAM2 first-frame pipeline.

    Returns:
        (keyframe, slam_map) — the reference keyframe and an empty map.
    """
    print("=" * 60)
    print("ORB-SLAM2 — Processing Frame 0 (Reference Frame)")
    print("=" * 60)

    # ── Load and convert to grayscale ──
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    print(f"\n📷 Loaded: {image_path}")
    print(f"   Size: {image.shape[1]}×{image.shape[0]} → grayscale")

    # ── Stage 1: Build image pyramid ──
    pyramid = build_image_pyramid(image, n_levels=8, scale_factor=1.2)
    print_pyramid_info(pyramid)

    # ── Stage 2: Detect FAST corners (demo on a small patch) ──
    print("\n⚡ FAST corner detection (manual demo on 100×100 patch):")
    patch = image[100:200, 100:200] if image.shape[0] > 200 else image[:100, :100]
    manual_corners = detect_fast_manual(patch, threshold=20, max_corners=50)
    print(f"   Detected {len(manual_corners)} corners in the patch (capped at 50 for demo)")
    print(f"   Example corners in patch: {manual_corners[:5]} (x, y)")

    # ── Stage 3: Extract full ORB features (OpenCV — production path) ──
    print("\n🎯 Full ORB extraction (OpenCV):")
    keypoints_cv, descriptors = extract_orb_features(
        image, n_features=10, n_levels=8, scale_factor=1.2,
    )

    if descriptors is None:
        print("   ⚠ Failed to extract descriptors. Image might be too small or blank.")
        return None, None

    # ── Stage 4: Analyze descriptors ──
    analyze_descriptors(descriptors, n_samples=5)

    # ── Stage 5: Orientation demo ──
    print("\n🧭 Orientation demo (manual intensity centroid on 3 keypoints):")
    for i in range(min(3, len(keypoints_cv))):
        kp = keypoints_cv[i]
        x, y = int(kp.pt[0]), int(kp.pt[1])
        manual_angle = compute_orientation(image, x, y, patch_radius=15)
        print(f"   Keypoint {i}: OpenCV angle={kp.angle:.1f}°, manual centroid={manual_angle:.1f}°")

    # ── Stage 6: Bag-of-words (placeholder) ──
    bow = compute_bow_placeholder(descriptors)

    # ── Stage 7: Create the reference keyframe ──
    keyframe = create_reference_keyframe(image, keypoints_cv, descriptors, pyramid)

    print(f"\n✅ Reference keyframe created:")
    print(f"   ID: {keyframe.id}")
    print(f"   Keypoints: {len(keyframe.keypoints)}")
    print(f"   Pose: Identity (world origin)")
    print(f"   Pyramid levels: {len(keyframe.pyramid)}")

    # ── Stage 8: Initialize the (empty) map ──
    slam_map = Map()
    slam_map.keyframes.append(keyframe)

    print(f"\n🗺️  Map state after frame 0:")
    print(f"   Keyframes: {len(slam_map.keyframes)}")
    print(f"   3D map points: {len(slam_map.map_points)}")
    print(f"   Status: NOT_INITIALIZED (need frame 2 to triangulate)")

    # ── Visualize ──
    visualize_results(image, keypoints_cv, pyramid)

    print("\n" + "=" * 60)
    print("Frame 0 complete. Waiting for camera to move...")
    print("=" * 60)

    return keyframe, slam_map


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORB-SLAM2 Step 1: Process first frame")
    parser.add_argument(
        "--frames_dir", type=str, default="KITTI/dataset/sequences/07/image_0",
        help="Directory containing frame images (0.png, 1.png, ...)",
    )
    parser.add_argument(
        "--frame", default="000000",
        help="Which frame number to process (default: 000000)",
    )
    args = parser.parse_args()

    image_path = os.path.join(args.frames_dir, f"{args.frame}.png")

    if not os.path.exists(image_path):
        # Try common alternative naming
        for alt in [f"frame_{args.frame:04d}.png", f"frame_{args.frame}.png", f"{args.frame:06d}.png"]:
            alt_path = os.path.join(args.frames_dir, alt)
            if os.path.exists(alt_path):
                image_path = alt_path
                break

    keyframe, slam_map = process_first_frame(image_path)