# ============================================================
# FACE VERIFICATION SERVICE
# ============================================================
#
# Uses InsightFace to:
# 1. Detect face in identity document
# 2. Extract document face embedding
# 3. Detect face from live webcam image
# 4. Extract live face embedding
# 5. Compare both embeddings
#
# Prototype threshold:
# 0.40
#
# ============================================================

import cv2
import numpy as np
import insightface

from insightface.app import FaceAnalysis


# ============================================================
# LOAD INSIGHTFACE MODEL
# ============================================================

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("========================================")
print("      INSIGHTFACE MODEL LOADED")
print("========================================")


# ============================================================
# GET LARGEST FACE
# ============================================================

def get_largest_face(faces):

    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda f: (
            f.bbox[2] - f.bbox[0]
        ) * (
            f.bbox[3] - f.bbox[1]
        )
    )


# ============================================================
# EXTRACT FACE EMBEDDING FROM IMAGE
# ============================================================

def extract_face_embedding(image):

    faces = face_app.get(image)

    if len(faces) == 0:

        return None, 0

    face = get_largest_face(faces)

    embedding = face.normed_embedding

    return embedding, len(faces)


# ============================================================
# DOCUMENT FACE VERIFICATION
# ============================================================

def prepare_document_face(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:

        return {
            "success": False,
            "error": "Document image could not be loaded."
        }

    embedding, face_count = (
        extract_face_embedding(image)
    )

    if embedding is None:

        return {
            "success": False,
            "error": "No face detected in the document."
        }

    return {
        "success": True,
        "embedding": embedding,
        "face_count": face_count
    }


# ============================================================
# LIVE FACE VERIFICATION
# ============================================================

def verify_live_face(
    document_embedding,
    live_image
):

    if document_embedding is None:

        return {
            "success": False,
            "error": "Document face embedding is missing."
        }

    if live_image is None:

        return {
            "success": False,
            "error": "Live image could not be loaded."
        }

    # --------------------------------------------------------
    # Detect live face
    # --------------------------------------------------------

    live_embedding, face_count = (
        extract_face_embedding(live_image)
    )

    if live_embedding is None:

        return {
            "success": False,
            "error": "No face detected in the live image.",
            "live_face": "Not Detected"
        }

    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    similarity = float(
        np.dot(
            document_embedding,
            live_embedding
        )
    )

    # --------------------------------------------------------
    # Prototype threshold
    # --------------------------------------------------------

    threshold = 0.40

    # --------------------------------------------------------
    # Determine result
    # --------------------------------------------------------

    if similarity >= threshold:

        result = "MATCH"
        identity_status = "VERIFIED"

    else:

        result = "NO MATCH"
        identity_status = "SUSPICIOUS"

    return {

        "success": True,

        "document_face": "Detected",

        "live_face": "Detected",

        "face_count": face_count,

        "similarity": round(
            similarity,
            3
        ),

        "threshold": threshold,

        "result": result,

        "identity_status": identity_status
    }