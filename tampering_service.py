# ============================================================
# DOCUMENT TAMPERING DETECTION SERVICE
# Supports: Image-based PAN / Aadhaar documents
# Method: Error Level Analysis (ELA)
# ============================================================

import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


# ------------------------------------------------------------
# 1. Create ELA image
# ------------------------------------------------------------

def create_ela_image(image_path, quality=90):
    """
    Creates an Error Level Analysis image.

    The original image is temporarily recompressed as JPEG.
    Differences between the original and recompressed image
    are amplified to reveal potentially suspicious areas.
    """

    original = Image.open(image_path).convert("RGB")

    temp_path = image_path + "_ela_temp.jpg"

    try:

        # Save recompressed copy
        original.save(
            temp_path,
            "JPEG",
            quality=quality
        )

        compressed = Image.open(temp_path).convert("RGB")

        # Calculate pixel difference
        ela = ImageChops.difference(
            original,
            compressed
        )

        # Find maximum difference
        extrema = ela.getextrema()

        max_difference = max(
            channel[1]
            for channel in extrema
        )

        if max_difference == 0:
            max_difference = 1

        # Amplify differences
        scale = 255.0 / max_difference

        ela = ImageEnhance.Brightness(
            ela
        ).enhance(scale)

        return original, ela

    finally:

        # Remove temporary ELA JPEG
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ------------------------------------------------------------
# 2. Main tampering analysis
# ------------------------------------------------------------

def analyze_tampering(image_path):

    original, ela = create_ela_image(
        image_path
    )

    # --------------------------------------------------------
    # Convert ELA image to grayscale
    # --------------------------------------------------------

    ela_np = np.array(ela)

    ela_gray = cv2.cvtColor(
        ela_np,
        cv2.COLOR_RGB2GRAY
    )

    # --------------------------------------------------------
    # Calculate ELA statistics
    # --------------------------------------------------------

    ela_mean = float(
        ela_gray.mean()
    )

    ela_max = int(
        ela_gray.max()
    )

    ela_std = float(
        ela_gray.std()
    )

    # --------------------------------------------------------
    # Create suspicious mask
    # --------------------------------------------------------

    threshold = 100

    _, suspicious_mask = cv2.threshold(
        ela_gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # Calculate suspicious area
    # --------------------------------------------------------

    total_pixels = suspicious_mask.size

    suspicious_pixels = np.count_nonzero(
        suspicious_mask
    )

    suspicious_percentage = (
        suspicious_pixels /
        total_pixels
    ) * 100

    # --------------------------------------------------------
    # Prototype risk score
    # --------------------------------------------------------

    risk_score = min(
        suspicious_percentage * 5,
        100
    )

    # --------------------------------------------------------
    # Determine status
    # --------------------------------------------------------

    if risk_score < 20:

        status = "LOW RISK"

    elif risk_score < 50:

        status = "MEDIUM RISK"

    else:

        status = "HIGH RISK"

    # --------------------------------------------------------
    # Find suspicious regions
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        suspicious_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious_regions = []

    min_area = 500

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Ignore small noise
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        suspicious_regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": float(area)
            }
        )

    # --------------------------------------------------------
    # Draw suspicious regions
    # --------------------------------------------------------

    result = np.array(
        original
    ).copy()

    for region in suspicious_regions:

        x = region["x"]
        y = region["y"]
        w = region["width"]
        h = region["height"]

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

    # --------------------------------------------------------
    # Save annotated result
    # --------------------------------------------------------

    result_filename = (
        "tampering_result.jpg"
    )

    result_path = os.path.join(
        "uploads",
        result_filename
    )

    result_bgr = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        result_path,
        result_bgr
    )

    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("           DOCUMENT TAMPERING ANALYSIS")
    print("=" * 60)

    print(
        f"Document Type       : IMAGE DOCUMENT"
    )

    print(
        f"ELA Mean            : "
        f"{ela_mean:.2f}"
    )

    print(
        f"ELA Maximum         : "
        f"{ela_max}"
    )

    print(
        f"ELA Std Deviation   : "
        f"{ela_std:.2f}"
    )

    print("-" * 60)

    print(
        f"Suspicious Area     : "
        f"{suspicious_percentage:.2f}%"
    )

    print(
        f"Suspicious Regions  : "
        f"{len(suspicious_regions)}"
    )

    print(
        f"Tampering Risk      : "
        f"{risk_score:.2f}/100"
    )

    print(
        f"Final Status        : "
        f"{status}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Return result to Flask
    # --------------------------------------------------------

    return {
        "ela_mean": round(
            ela_mean,
            2
        ),

        "ela_max": ela_max,

        "ela_std": round(
            ela_std,
            2
        ),

        "suspicious_percentage": round(
            suspicious_percentage,
            2
        ),

        "suspicious_regions": suspicious_regions,

        "region_count": len(
            suspicious_regions
        ),

        "risk_score": round(
            risk_score,
            2
        ),

        "status": status,

        "result_image": "/uploads/tampering_result.jpg"
    }