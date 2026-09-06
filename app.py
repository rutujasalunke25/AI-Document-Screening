from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
    session
)

import sqlite3
import os
import base64
import numpy as np
import cv2

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename
from ocr_service import process_document
from validation_service import validate_document
from tampering_service import analyze_tampering
from face_service import prepare_document_face


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Secret key is required for Flask session data.
app.secret_key = "ai-document-screening-secret-key-change-this"

# ============================================================
# OVERALL RISK SCORING
# ============================================================

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _calculate_ocr_risk(result):
    """Estimate OCR risk from missing expected fields."""
    if not result:
        return 100.0

    document_type = str(
        result.get("document_type", "")
    ).upper().strip()

    if document_type == "PAN":
        fields = [
            result.get("name"),
            result.get("dob"),
            result.get("gender"),
            result.get("pan_number")
        ]

    elif document_type == "AADHAAR":
        fields = [
            result.get("name"),
            result.get("dob"),
            result.get("gender"),
            result.get("aadhaar_number")
        ]

    elif document_type == "PASSPORT":
        fields = [
            result.get("passport_number"),
            result.get("surname"),
            result.get("given_name"),
            result.get("nationality"),
            result.get("dob"),
            result.get("sex") or result.get("gender"),
            result.get("place_of_birth"),
            result.get("place_of_issue"),
            result.get("date_of_issue"),
            result.get("date_of_expiry")
        ]

    else:
        return 80.0

    total = len(fields)

    if total == 0:
        return 100.0

    detected = sum(
        1
        for field in fields
        if field is not None
        and str(field).strip() != ""
    )

    return round(
        ((total - detected) / total) * 100.0,
        2
    )


def _calculate_validation_risk(validation):
    """Convert validation failures into a 0-100 risk."""
    if not validation:
        return 100.0

    checks = validation.get("checks", {})

    if not checks:
        return 100.0

    total = len(checks)

    failed = sum(
        1
        for status in checks.values()
        if str(status).upper() != "PASS"
    )

    return round(
        (failed / total) * 100.0,
        2
    )


def _calculate_tampering_risk(tampering):
    """Use the tampering service's existing risk score."""
    if not tampering:
        return 50.0

    if str(
        tampering.get("status", "")
    ).upper() == "ERROR":
        return 60.0

    risk = _safe_float(
        tampering.get("risk_score"),
        50.0
    )

    return round(
        max(0.0, min(100.0, risk)),
        2
    )


def _calculate_face_risk(
    face_result=None,
    face_verification=None
):
    """
    Face risk:
      VERIFIED     -> 0
      NO MATCH     -> 100
      document face detected but not live verified -> 5
    """

    if face_verification:

        identity_status = str(
            face_verification.get(
                "identity_status",
                ""
            )
        ).upper().strip()

        verification_result = str(
            face_verification.get(
                "result",
                ""
            )
        ).upper().strip()

        if identity_status == "VERIFIED":
            return 0.0

        if verification_result in (
            "NO MATCH",
            "NOT MATCHED",
            "MISMATCH"
        ):
            return 100.0

        try:
            similarity = float(
                face_verification.get(
                    "similarity",
                    0
                )
            )

            threshold = float(
                face_verification.get(
                    "threshold",
                    0.40
                )
            )

            if threshold > 0:
                if similarity >= threshold:
                    return 0.0

                risk = (
                    1.0
                    - (similarity / threshold)
                ) * 100.0

                return round(
                    max(
                        0.0,
                        min(100.0, risk)
                    ),
                    2
                )

        except (
            TypeError,
            ValueError,
            ZeroDivisionError
        ):
            pass

        return 60.0

    if face_result and face_result.get("success"):
        return 5.0

    return 80.0


def _get_risk_level(score):
    if score <= 20:
        return "LOW RISK"
    if score <= 50:
        return "MEDIUM RISK"
    if score <= 75:
        return "HIGH RISK"
    return "VERY HIGH RISK"


def _generate_risk_comment(
    score,
    ocr_risk,
    validation_risk,
    tampering_risk,
    face_risk,
    face_verified=False
):
    reasons = []

    if ocr_risk >= 50:
        reasons.append(
            "OCR extraction has missing or incomplete information"
        )
    elif ocr_risk >= 20:
        reasons.append(
            "some document information could not be confidently extracted"
        )

    if validation_risk >= 50:
        reasons.append(
            "document validation checks failed"
        )
    elif validation_risk > 0:
        reasons.append(
            "some document validation checks require attention"
        )

    if tampering_risk >= 75:
        reasons.append(
            "strong indicators of possible document tampering were detected"
        )
    elif tampering_risk >= 40:
        reasons.append(
            "some suspicious document regions were detected"
        )

    if not face_verified:
        if face_risk >= 75:
            reasons.append(
                "the document face could not be successfully verified"
            )
        elif face_risk >= 30:
            reasons.append(
                "face verification requires attention"
            )

    if score <= 20:
        return (
            "Document appears genuine with no significant "
            "inconsistencies found."
        )

    if score <= 50:
        if reasons:
            return (
                "Document shows some suspicious indicators: "
                + "; ".join(reasons)
                + ". Further verification is recommended."
            )
        return (
            "Document contains some inconsistencies. "
            "Further verification is recommended."
        )

    if score <= 75:
        if reasons:
            return (
                "Document requires further verification because "
                + "; ".join(reasons)
                + "."
            )
        return (
            "Several screening indicators are suspicious. "
            "Further verification is strongly recommended."
        )

    if reasons:
        return (
            "Multiple verification checks indicate a high "
            "possibility of document or identity irregularities: "
            + "; ".join(reasons)
            + "."
        )

    return (
        "Multiple screening checks produced strong risk indicators. "
        "The document should undergo manual verification."
    )


def calculate_overall_risk(
    result,
    validation,
    tampering,
    face_result=None,
    face_verification=None
):
    """
    Weighted overall fake-risk estimate.

    OCR          = 20%
    Validation   = 25%
    Tampering    = 30%
    Face         = 25%
    """

    ocr_risk = _calculate_ocr_risk(result)
    validation_risk = _calculate_validation_risk(validation)
    tampering_risk = _calculate_tampering_risk(tampering)
    face_risk = _calculate_face_risk(
        face_result,
        face_verification
    )

    score = (
        (ocr_risk * 0.20)
        + (validation_risk * 0.25)
        + (tampering_risk * 0.30)
        + (face_risk * 0.25)
    )

    score = round(
        max(0.0, min(100.0, score)),
        2
    )

    face_verified = bool(
        face_verification
        and str(
            face_verification.get(
                "identity_status",
                ""
            )
        ).upper().strip()
        == "VERIFIED"
    )

    return {
        "risk_score": score,
        "risk_level": _get_risk_level(score),
        "comment": _generate_risk_comment(
            score,
            ocr_risk,
            validation_risk,
            tampering_risk,
            face_risk,
            face_verified
        ),
        "components": {
            "ocr": ocr_risk,
            "validation": validation_risk,
            "tampering": tampering_risk,
            "face": face_risk
        }
    }


def _print_overall_risk(overall_risk):
    print()
    print("=" * 50)
    print("             OVERALL RISK ASSESSMENT")
    print("=" * 50)
    print(
        f"Fake Risk Score : "
        f"{overall_risk['risk_score']:.2f}%"
    )
    print(
        f"Risk Level      : "
        f"{overall_risk['risk_level']}"
    )
    print("-" * 50)
    print(
        f"OCR Risk        : "
        f"{overall_risk['components']['ocr']:.2f}%"
    )
    print(
        f"Validation Risk : "
        f"{overall_risk['components']['validation']:.2f}%"
    )
    print(
        f"Tampering Risk  : "
        f"{overall_risk['components']['tampering']:.2f}%"
    )
    print(
        f"Face Risk       : "
        f"{overall_risk['components']['face']:.2f}%"
    )
    print("-" * 50)
    print(
        "Comment         : "
        + overall_risk["comment"]
    )
    print("=" * 50)



# ============================================================
# DATABASE
# ============================================================

def get_db():

    conn = sqlite3.connect("database.db")

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# CREATE TABLE
# ============================================================

def create_table():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            mobile TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            organization TEXT NOT NULL,

            password TEXT NOT NULL

        )
    """)

    conn.commit()

    conn.close()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return redirect(
        url_for("login")
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)

def signup():

    if request.method == "POST":

        name = request.form["name"]

        mobile = request.form["mobile"]

        email = request.form["email"]

        organization = request.form["organization"]

        password = request.form["password"]


        hashed_password = generate_password_hash(
            password
        )


        try:

            conn = get_db()


            conn.execute("""
                INSERT INTO users
                (
                    name,
                    mobile,
                    email,
                    organization,
                    password
                )

                VALUES (?, ?, ?, ?, ?)
            """,
            (
                name,
                mobile,
                email,
                organization,
                hashed_password
            ))


            conn.commit()

            conn.close()


            return redirect(
                url_for("login")
            )


        except sqlite3.IntegrityError:

            return "Email already registered!"


    return render_template(
        "signup.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)

def login():

    if request.method == "POST":

        email = request.form["email"]

        password = request.form["password"]


        conn = get_db()


        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()


        conn.close()


        if user and check_password_hash(
            user["password"],
            password
        ):

            return redirect(
                url_for("dashboard")
            )


        return "Invalid Email or Password!"


    return render_template(
        "login.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        "uploads",
        filename
    )

# ============================================================
# UPLOAD + OCR + VALIDATION
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)

def upload():

    file = request.files.get(
        "document"
    )


    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not file:

        return "No document selected!"


    if file.filename == "":

        return "No document selected!"


    # --------------------------------------------------------
    # CREATE UPLOAD FOLDER
    # --------------------------------------------------------

    upload_folder = "uploads"

    os.makedirs(
        upload_folder,
        exist_ok=True
    )


    # --------------------------------------------------------
    # SECURE FILE NAME
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    filepath = os.path.join(
        upload_folder,
        filename
    )


    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    file.save(
        filepath
    )


    # ========================================================
    # PREPARE DOCUMENT FACE
    # ========================================================

    face_result = None

    image_extensions = [
    ".jpg",
    ".jpeg",
    ".png"
]

    extension = os.path.splitext(
    filepath
)[1].lower()

    if extension in image_extensions:

        try:

            face_result = prepare_document_face(
            filepath
        )

        except Exception as e:

            print(
            "Face detection failed:",
            e
        )

            face_result = {
            "success": False,
            "error": str(e)
        }

    # ========================================================
    # OCR
    # ========================================================

    try:

        result = process_document(filepath)

        print()
        print("========== OCR RESULT FROM process_document ==========")
        print(result)
        print("======================================================")

    except Exception as e:

        return (
            "OCR processing failed: "
            + str(e)
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    try:

        validation_result = validate_document(
            result
        )

    except Exception as e:

        return (
            "Document validation failed: "
            + str(e)
        )

    # ========================================================
    # TAMPERING DETECTION
    # ========================================================

    tampering_result = None

    # ELA requires an image
    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png"
    ]

    extension = os.path.splitext(
        filepath
    )[1].lower()

    if extension in image_extensions:

        try:

            tampering_result = analyze_tampering(
                filepath
            )

        except Exception as e:

            print(
                "Tampering detection failed:",
                e
            )

            tampering_result = {
                "status": "ERROR",
                "error": str(e)
            }


    # ========================================================
    # OVERALL RISK SCORE
    # ========================================================

    overall_risk = calculate_overall_risk(
        result=result,
        validation=validation_result,
        tampering=tampering_result,
        face_result=face_result
    )

    _print_overall_risk(overall_risk)

    # Store only small JSON-safe values needed when the
    # live face verification is performed later.
    session["risk_base"] = {
        "ocr": overall_risk["components"]["ocr"],
        "validation": overall_risk["components"]["validation"],
        "tampering": overall_risk["components"]["tampering"],
        "filename": filename
    }

    # ========================================================
    # SEND RESULTS TO DASHBOARD
    # ========================================================

    return render_template(
        "dashboard.html",
        result=result,
        validation=validation_result,
        tampering=tampering_result,
        face_result=face_result,
        overall_risk=overall_risk,
        document_filename=filename
    )

# ============================================================
# FACE VERIFICATION
# ============================================================

@app.route(
    "/verify-face",
    methods=["POST"]
)
def verify_face():

    data = request.get_json()

    if not data:

        return jsonify({
            "success": False,
            "error": "No image received."
        })

    image_data = data.get(
        "image"
    )

    if not image_data:

        return jsonify({
            "success": False,
            "error": "Camera image missing."
        })

    # --------------------------------------------------------
    # Remove base64 header
    # --------------------------------------------------------

    try:

        if "," in image_data:

            image_data = (
                image_data.split(
                    ",",
                    1
                )[1]
            )

        image_bytes = base64.b64decode(
            image_data
        )

    except Exception:

        return jsonify({
            "success": False,
            "error": "Invalid camera image."
        })

    # --------------------------------------------------------
    # Convert bytes → OpenCV image
    # --------------------------------------------------------

    try:

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        live_image = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

    except Exception:

        return jsonify({
            "success": False,
            "error": "Could not process camera image."
        })

    if live_image is None:

        return jsonify({
            "success": False,
            "error": "Camera image could not be decoded."
        })

    # --------------------------------------------------------
    # Get document filename
    # --------------------------------------------------------

    document_filename = data.get(
        "document_filename"
    )

    if not document_filename:

        return jsonify({
            "success": False,
            "error": "Document reference missing."
        })

    document_path = os.path.join(
        "uploads",
        secure_filename(
            document_filename
        )
    )

    if not os.path.exists(
        document_path
    ):

        return jsonify({
            "success": False,
            "error": "Uploaded document not found."
        })

    # --------------------------------------------------------
    # Extract document face
    # --------------------------------------------------------

    try:

        document_result = (
            prepare_document_face(
                document_path
            )
        )

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })

    if not document_result.get(
        "success"
    ):

        return jsonify({
            "success": False,
            "error": document_result.get(
                "error",
                "Document face not detected."
            )
        })

    document_embedding = (
        document_result["embedding"]
    )

    # --------------------------------------------------------
    # Verify live face
    # --------------------------------------------------------

    try:

        from face_service import (
            verify_live_face
        )

        verification = verify_live_face(
            document_embedding,
            live_image
        )

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print("\n")
    print("=" * 40)
    print("       FACE VERIFICATION RESULT")
    print("=" * 40)

    print(
        f"Document Face     : "
        f"{verification.get('document_face', 'Not Detected')}"
    )

    print(
        f"Live Face         : "
        f"{verification.get('live_face', 'Not Detected')}"
    )

    print(
        f"Similarity Score  : "
        f"{verification.get('similarity', 0):.2f}"
    )

    print(
        f"Threshold         : "
        f"{verification.get('threshold', 0.40):.2f}"
    )

    print(
        f"Result            : "
        f"{verification.get('result', 'NO MATCH')}"
    )

    print(
        f"Identity Status   : "
        f"{verification.get('identity_status', 'UNVERIFIED')}"
    )

    print("=" * 40)

    # --------------------------------------------------------
    # UPDATE OVERALL RISK AFTER LIVE FACE VERIFICATION
    # --------------------------------------------------------

    risk_base = session.get("risk_base", {})

    if risk_base:

        # Reconstruct the three already calculated component risks.
        # The face component is replaced with the real live result.
        ocr_risk = _safe_float(
            risk_base.get("ocr"),
            50
        )
        validation_risk = _safe_float(
            risk_base.get("validation"),
            50
        )
        tampering_risk = _safe_float(
            risk_base.get("tampering"),
            50
        )
        face_risk = _calculate_face_risk(
            face_verification=verification
        )

        overall_score = round(
            max(
                0.0,
                min(
                    100.0,
                    (ocr_risk * 0.20)
                    + (validation_risk * 0.25)
                    + (tampering_risk * 0.30)
                    + (face_risk * 0.25)
                )
            ),
            2
        )

        face_verified = (
            str(
                verification.get(
                    "identity_status",
                    ""
                )
            ).upper().strip()
            == "VERIFIED"
        )

        overall_risk = {
            "risk_score": overall_score,
            "risk_level": _get_risk_level(
                overall_score
            ),
            "comment": _generate_risk_comment(
                overall_score,
                ocr_risk,
                validation_risk,
                tampering_risk,
                face_risk,
                face_verified
            ),
            "components": {
                "ocr": round(ocr_risk, 2),
                "validation": round(
                    validation_risk,
                    2
                ),
                "tampering": round(
                    tampering_risk,
                    2
                ),
                "face": round(
                    face_risk,
                    2
                )
            }
        }

        _print_overall_risk(
            overall_risk
        )

        verification["overall_risk"] = (
            overall_risk
        )

    return jsonify(verification)
# ============================================================
# FORGOT PASSWORD
# ============================================================

@app.route("/forgot-password")
def forgot_password():

    return (
        "Forgot Password feature coming soon."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    create_table()
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
    session
)

import sqlite3
import os
import base64
import numpy as np
import cv2

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from ocr_service import process_document
from validation_service import validate_document
from tampering_service import analyze_tampering
from face_service import prepare_document_face


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Required for storing the calculated risk components between
# document upload and live face verification.
app.secret_key = "ai-document-screening-secret-key-change-this"


# ============================================================
# OVERALL RISK SCORING
# ============================================================

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _calculate_ocr_risk(result):
    """
    Estimate OCR risk from missing expected fields.

    More missing OCR fields = higher risk.
    """

    if not result:
        return 100.0

    document_type = str(
        result.get("document_type", "")
    ).upper().strip()

    if document_type == "PAN":
        fields = [
            result.get("name"),
            result.get("dob"),
            result.get("gender"),
            result.get("pan_number")
        ]

    elif document_type == "AADHAAR":
        fields = [
            result.get("name"),
            result.get("dob"),
            result.get("gender"),
            result.get("aadhaar_number")
        ]

    elif document_type == "PASSPORT":
        fields = [
            result.get("passport_number"),
            result.get("surname"),
            result.get("given_name"),
            result.get("nationality"),
            result.get("dob"),
            result.get("sex") or result.get("gender"),
            result.get("place_of_birth"),
            result.get("place_of_issue"),
            result.get("date_of_issue"),
            result.get("date_of_expiry")
        ]

    else:
        return 80.0

    total = len(fields)

    if total == 0:
        return 100.0

    detected = sum(
        1
        for field in fields
        if field is not None
        and str(field).strip() != ""
    )

    return round(
        ((total - detected) / total) * 100.0,
        2
    )


def _calculate_validation_risk(validation):
    """
    Convert validation failures into a 0-100 risk.
    """

    if not validation:
        return 100.0

    checks = validation.get("checks", {})

    if not checks:
        return 100.0

    total = len(checks)

    failed = sum(
        1
        for status in checks.values()
        if str(status).upper().strip() != "PASS"
    )

    return round(
        (failed / total) * 100.0,
        2
    )


def _calculate_tampering_risk(tampering):
    """
    Use the tampering service's existing risk score.
    """

    if not tampering:
        return 50.0

    if str(
        tampering.get("status", "")
    ).upper().strip() == "ERROR":
        return 60.0

    risk = _safe_float(
        tampering.get("risk_score"),
        50.0
    )

    return round(
        max(0.0, min(100.0, risk)),
        2
    )


def _calculate_face_risk(
    face_result=None,
    face_verification=None
):
    """
    Calculate the face component of the overall fake-risk score.

    IMPORTANT:
    Detecting a face inside the uploaded document is NOT the same
    as verifying the live person.

    States:

    1. Live face verified and matched:
       Face risk = 0%

    2. Live face verification produced a mismatch:
       Face risk = 100%

    3. Live face verification produced a usable similarity score:
       Calculate risk from similarity and threshold.

    4. Document face exists, but live verification has NOT happened:
       Face risk = 50% (PENDING / NEUTRAL)

    5. Document face could not be detected:
       Face risk = 80% because identity verification cannot be completed.
    """

    # --------------------------------------------------------
    # LIVE FACE VERIFICATION HAS HAPPENED
    # --------------------------------------------------------

    if face_verification:

        identity_status = str(
            face_verification.get(
                "identity_status",
                ""
            )
        ).upper().strip()

        verification_result = str(
            face_verification.get(
                "result",
                ""
            )
        ).upper().strip()

        # Exact successful identity verification.
        if identity_status in (
            "VERIFIED",
            "MATCHED",
            "IDENTITY VERIFIED"
        ):
            return 0.0

        # Exact mismatch states.
        if verification_result in (
            "NO MATCH",
            "NOT MATCHED",
            "MISMATCH",
            "FAILED",
            "FAIL"
        ) or identity_status in (
            "NOT VERIFIED",
            "MISMATCH",
            "NO MATCH"
        ):
            return 100.0

        # ----------------------------------------------------
        # Similarity-based fallback
        # ----------------------------------------------------

        similarity_value = face_verification.get(
            "similarity"
        )

        threshold_value = face_verification.get(
            "threshold",
            0.40
        )

        try:
            similarity = float(similarity_value)
            threshold = float(threshold_value)

            if threshold > 0:

                similarity = max(
                    0.0,
                    min(1.0, similarity)
                )

                threshold = max(
                    0.000001,
                    min(1.0, threshold)
                )

                # Similarity at/above threshold means match.
                if similarity >= threshold:
                    return 0.0

                # Below threshold:
                # 0 similarity = 100 risk
                # threshold similarity = 0 risk
                risk = (
                    1.0
                    - (similarity / threshold)
                ) * 100.0

                return round(
                    max(
                        0.0,
                        min(100.0, risk)
                    ),
                    2
                )

        except (
            TypeError,
            ValueError,
            ZeroDivisionError
        ):
            pass

        # A verification request happened but no usable result
        # could be interpreted safely.
        return 80.0

    # --------------------------------------------------------
    # LIVE VERIFICATION HAS NOT HAPPENED
    # --------------------------------------------------------

    if face_result and face_result.get("success"):
        # Face detected in the document, but the person has not
        # yet been verified using the camera.
        return 50.0

    # No document face means the identity check cannot currently
    # be completed.
    return 80.0


def _get_risk_level(score):

    if score <= 20:
        return "LOW RISK"

    if score <= 50:
        return "MEDIUM RISK"

    if score <= 75:
        return "HIGH RISK"

    return "VERY HIGH RISK"


def _generate_risk_comment(
    score,
    ocr_risk,
    validation_risk,
    tampering_risk,
    face_risk,
    face_verified=False
):
    """
    Generate a comment that reflects the ACTUAL state of all
    screening components.

    A document must NOT be described as fully genuine while
    live face verification is still pending.
    """

    reasons = []

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    if ocr_risk >= 50:
        reasons.append(
            "OCR extraction has missing or incomplete information"
        )

    elif ocr_risk >= 20:
        reasons.append(
            "some document information could not be confidently extracted"
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if validation_risk >= 50:
        reasons.append(
            "document validation checks failed"
        )

    elif validation_risk > 0:
        reasons.append(
            "some document validation checks require attention"
        )

    # --------------------------------------------------------
    # TAMPERING
    # --------------------------------------------------------

    if tampering_risk >= 75:
        reasons.append(
            "strong indicators of possible document tampering were detected"
        )

    elif tampering_risk >= 40:
        reasons.append(
            "some suspicious document regions were detected"
        )

    # --------------------------------------------------------
    # FACE VERIFICATION
    # --------------------------------------------------------

    if not face_verified:

        if face_risk >= 75:
            reasons.append(
                "the document face could not be successfully verified"
            )

        elif face_risk >= 30:
            reasons.append(
                "live face verification is still pending"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # Do NOT return the old "genuine" comment here.
        # Face verification has not happened yet.
        # ----------------------------------------------------

        if reasons:
            return (
                "Preliminary screening result: "
                + "; ".join(reasons)
                + ". Complete live face verification before making "
                "a final authenticity decision."
            )

        return (
            "Preliminary document screening is complete, but live "
            "face verification is still pending. Complete face "
            "verification before making a final authenticity decision."
        )

    # --------------------------------------------------------
    # FACE VERIFIED
    # --------------------------------------------------------

    if face_risk >= 75:
        reasons.append(
            "the live face did not match the document face"
        )

    # --------------------------------------------------------
    # FINAL VERIFIED COMMENT
    # --------------------------------------------------------

    if score <= 20:

        if not reasons:
            return (
                "Document appears genuine with no significant "
                "inconsistencies found, and the live face matches "
                "the document face."
            )

    if score <= 50:

        if reasons:
            return (
                "Document shows some suspicious indicators: "
                + "; ".join(reasons)
                + ". Further verification is recommended."
            )

        return (
            "Document screening indicators are relatively low-risk "
            "and the live face matches the document face."
        )

    if score <= 75:

        if reasons:
            return (
                "Document requires further verification because "
                + "; ".join(reasons)
                + "."
            )

        return (
            "Several screening indicators are suspicious. "
            "Further verification is strongly recommended."
        )

    if reasons:
        return (
            "Multiple verification checks indicate a high "
            "possibility of document or identity irregularities: "
            + "; ".join(reasons)
            + "."
        )

    return (
        "Multiple screening checks produced strong risk indicators. "
        "The document should undergo manual verification."
    )


def calculate_overall_risk(
    result,
    validation,
    tampering,
    face_result=None,
    face_verification=None
):
    """
    Weighted overall fake-risk estimate.

    OCR          = 20%
    Validation   = 25%
    Tampering    = 30%
    Face         = 25%
    """

    ocr_risk = _calculate_ocr_risk(
        result
    )

    validation_risk = _calculate_validation_risk(
        validation
    )

    tampering_risk = _calculate_tampering_risk(
        tampering
    )

    face_risk = _calculate_face_risk(
        face_result=face_result,
        face_verification=face_verification
    )

    score = (
        (ocr_risk * 0.20)
        + (validation_risk * 0.25)
        + (tampering_risk * 0.30)
        + (face_risk * 0.25)
    )

    score = round(
        max(
            0.0,
            min(100.0, score)
        ),
        2
    )

    face_verified = bool(
        face_verification
        and str(
            face_verification.get(
                "identity_status",
                ""
            )
        ).upper().strip()
        in (
            "VERIFIED",
            "MATCHED",
            "IDENTITY VERIFIED"
        )
    )

    return {
        "risk_score": score,

        "risk_level": _get_risk_level(
            score
        ),

        "comment": _generate_risk_comment(
            score=score,
            ocr_risk=ocr_risk,
            validation_risk=validation_risk,
            tampering_risk=tampering_risk,
            face_risk=face_risk,
            face_verified=face_verified
        ),

        "components": {
            "ocr": round(
                ocr_risk,
                2
            ),

            "validation": round(
                validation_risk,
                2
            ),

            "tampering": round(
                tampering_risk,
                2
            ),

            "face": round(
                face_risk,
                2
            )
        },

        # Useful for the dashboard to know whether this is a
        # preliminary score or a final post-face-verification score.
        "face_verified": face_verified
    }


def _print_overall_risk(overall_risk):

    print()
    print("=" * 50)
    print("             OVERALL RISK ASSESSMENT")
    print("=" * 50)

    print(
        f"Fake Risk Score : "
        f"{overall_risk['risk_score']:.2f}%"
    )

    print(
        f"Risk Level      : "
        f"{overall_risk['risk_level']}"
    )

    print("-" * 50)

    print(
        f"OCR Risk        : "
        f"{overall_risk['components']['ocr']:.2f}%"
    )

    print(
        f"Validation Risk : "
        f"{overall_risk['components']['validation']:.2f}%"
    )

    print(
        f"Tampering Risk  : "
        f"{overall_risk['components']['tampering']:.2f}%"
    )

    print(
        f"Face Risk       : "
        f"{overall_risk['components']['face']:.2f}%"
    )

    print(
        f"Face Verified   : "
        f"{overall_risk.get('face_verified', False)}"
    )

    print("-" * 50)

    print(
        "Comment         : "
        + overall_risk["comment"]
    )

    print("=" * 50)


# ============================================================
# DATABASE
# ============================================================

def get_db():

    conn = sqlite3.connect(
        "database.db"
    )

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# CREATE TABLE
# ============================================================

def create_table():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            mobile TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            organization TEXT NOT NULL,

            password TEXT NOT NULL

        )
    """)

    conn.commit()

    conn.close()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return redirect(
        url_for("login")
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        name = request.form["name"]

        mobile = request.form["mobile"]

        email = request.form["email"]

        organization = request.form["organization"]

        password = request.form["password"]

        hashed_password = generate_password_hash(
            password
        )

        try:

            conn = get_db()

            conn.execute(
                """
                INSERT INTO users
                (
                    name,
                    mobile,
                    email,
                    organization,
                    password
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name,
                    mobile,
                    email,
                    organization,
                    hashed_password
                )
            )

            conn.commit()

            conn.close()

            return redirect(
                url_for("login")
            )

        except sqlite3.IntegrityError:

            return "Email already registered!"

    return render_template(
        "signup.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form["email"]

        password = request.form["password"]

        conn = get_db()

        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        conn.close()

        if user and check_password_hash(
            user["password"],
            password
        ):

            return redirect(
                url_for("dashboard")
            )

        return "Invalid Email or Password!"

    return render_template(
        "login.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        result={},
        validation={},
        tampering={},
        face_result={},
        overall_risk=None,
        document_filename=""
    )


# ============================================================
# UPLOADED FILES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        "uploads",
        filename
    )


# ============================================================
# UPLOAD + OCR + VALIDATION + TAMPERING + FACE
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    file = request.files.get(
        "document"
    )

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not file:

        return "No document selected!"

    if file.filename == "":

        return "No document selected!"

    # --------------------------------------------------------
    # CREATE UPLOAD FOLDER
    # --------------------------------------------------------

    upload_folder = "uploads"

    os.makedirs(
        upload_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SECURE FILE NAME
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        upload_folder,
        filename
    )

    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    file.save(
        filepath
    )

    # ========================================================
    # PREPARE DOCUMENT FACE
    # ========================================================

    face_result = None

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png"
    ]

    extension = os.path.splitext(
        filepath
    )[1].lower()

    if extension in image_extensions:

        try:

            face_result = prepare_document_face(
                filepath
            )

        except Exception as e:

            print(
                "Face detection failed:",
                e
            )

            face_result = {
                "success": False,
                "error": str(e)
            }

    # ========================================================
    # OCR
    # ========================================================

    try:

        result = process_document(
            filepath
        )

        print()
        print(
            "========== OCR RESULT FROM process_document =========="
        )
        print(result)
        print(
            "======================================================"
        )

    except Exception as e:

        return (
            "OCR processing failed: "
            + str(e)
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    try:

        validation_result = validate_document(
            result
        )

    except Exception as e:

        return (
            "Document validation failed: "
            + str(e)
        )

    # ========================================================
    # TAMPERING DETECTION
    # ========================================================

    tampering_result = None

    if extension in image_extensions:

        try:

            tampering_result = analyze_tampering(
                filepath
            )

        except Exception as e:

            print(
                "Tampering detection failed:",
                e
            )

            tampering_result = {
                "status": "ERROR",
                "error": str(e)
            }

    # ========================================================
    # INITIAL OVERALL RISK SCORE
    # ========================================================

    # No live face verification has happened yet.
    # Therefore face risk is deliberately PENDING / 50%.
    overall_risk = calculate_overall_risk(
        result=result,
        validation=validation_result,
        tampering=tampering_result,
        face_result=face_result,
        face_verification=None
    )

    _print_overall_risk(
        overall_risk
    )

    # --------------------------------------------------------
    # Store only the small JSON-safe component values required
    # to recalculate the overall score after live verification.
    # --------------------------------------------------------

    session["risk_base"] = {
        "ocr": overall_risk["components"]["ocr"],
        "validation": overall_risk["components"]["validation"],
        "tampering": overall_risk["components"]["tampering"],
        "filename": filename
    }

    # Explicitly clear any old verification state from a
    # previous document.
    session.pop(
        "face_verification",
        None
    )

    # ========================================================
    # SEND RESULTS TO DASHBOARD
    # ========================================================

    return render_template(
        "dashboard.html",
        result=result,
        validation=validation_result,
        tampering=tampering_result,
        face_result=face_result,
        overall_risk=overall_risk,
        document_filename=filename
    )


# ============================================================
# FACE VERIFICATION
# ============================================================

@app.route(
    "/verify-face",
    methods=["POST"]
)
def verify_face():

    data = request.get_json()

    if not data:

        return jsonify({
            "success": False,
            "error": "No image received."
        })

    image_data = data.get(
        "image"
    )

    if not image_data:

        return jsonify({
            "success": False,
            "error": "Camera image missing."
        })

    # --------------------------------------------------------
    # Remove base64 header
    # --------------------------------------------------------

    try:

        if "," in image_data:

            image_data = (
                image_data.split(
                    ",",
                    1
                )[1]
            )

        image_bytes = base64.b64decode(
            image_data
        )

    except Exception:

        return jsonify({
            "success": False,
            "error": "Invalid camera image."
        })

    # --------------------------------------------------------
    # Convert bytes -> OpenCV image
    # --------------------------------------------------------

    try:

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        live_image = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

    except Exception:

        return jsonify({
            "success": False,
            "error": "Could not process camera image."
        })

    if live_image is None:

        return jsonify({
            "success": False,
            "error": "Camera image could not be decoded."
        })

    # --------------------------------------------------------
    # Get document filename
    # --------------------------------------------------------

    document_filename = data.get(
        "document_filename"
    )

    if not document_filename:

        return jsonify({
            "success": False,
            "error": "Document reference missing."
        })

    document_filename = secure_filename(
        document_filename
    )

    document_path = os.path.join(
        "uploads",
        document_filename
    )

    if not os.path.exists(
        document_path
    ):

        return jsonify({
            "success": False,
            "error": "Uploaded document not found."
        })

    # --------------------------------------------------------
    # Extract document face
    # --------------------------------------------------------

    try:

        document_result = prepare_document_face(
            document_path
        )

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })

    if not document_result.get(
        "success"
    ):

        return jsonify({
            "success": False,
            "error": document_result.get(
                "error",
                "Document face not detected."
            )
        })

    document_embedding = (
        document_result["embedding"]
    )

    # --------------------------------------------------------
    # Verify live face
    # --------------------------------------------------------

    try:

        from face_service import verify_live_face

        verification = verify_live_face(
            document_embedding,
            live_image
        )

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print()
    print("=" * 40)
    print("       FACE VERIFICATION RESULT")
    print("=" * 40)

    print(
        f"Document Face     : "
        f"{verification.get('document_face', 'Not Detected')}"
    )

    print(
        f"Live Face         : "
        f"{verification.get('live_face', 'Not Detected')}"
    )

    print(
        f"Similarity Score  : "
        f"{verification.get('similarity', 0):.2f}"
    )

    print(
        f"Threshold         : "
        f"{verification.get('threshold', 0.40):.2f}"
    )

    print(
        f"Result            : "
        f"{verification.get('result', 'NO MATCH')}"
    )

    print(
        f"Identity Status   : "
        f"{verification.get('identity_status', 'UNVERIFIED')}"
    )

    print("=" * 40)

    # ========================================================
    # RECALCULATE OVERALL RISK USING THE REAL LIVE FACE RESULT
    # ========================================================

    risk_base = session.get(
        "risk_base",
        {}
    )

    if not risk_base:

        return jsonify({
            **verification,
            "overall_risk": None,
            "warning": (
                "Face verification completed, but the base "
                "document risk data is unavailable. Please "
                "upload the document again."
            )
        })

    # --------------------------------------------------------
    # Recover the three component scores calculated at upload.
    # --------------------------------------------------------

    ocr_risk = _safe_float(
        risk_base.get("ocr"),
        50.0
    )

    validation_risk = _safe_float(
        risk_base.get("validation"),
        50.0
    )

    tampering_risk = _safe_float(
        risk_base.get("tampering"),
        50.0
    )

    # --------------------------------------------------------
    # Calculate the REAL face risk from the live verification.
    # --------------------------------------------------------

    face_risk = _calculate_face_risk(
        face_verification=verification
    )

    # --------------------------------------------------------
    # Calculate final weighted score.
    # --------------------------------------------------------

    overall_score = round(
        max(
            0.0,
            min(
                100.0,
                (ocr_risk * 0.20)
                + (validation_risk * 0.25)
                + (tampering_risk * 0.30)
                + (face_risk * 0.25)
            )
        ),
        2
    )

    face_verified = (
        str(
            verification.get(
                "identity_status",
                ""
            )
        ).upper().strip()
        in (
            "VERIFIED",
            "MATCHED",
            "IDENTITY VERIFIED"
        )
    )

    overall_risk = {
        "risk_score": overall_score,

        "risk_level": _get_risk_level(
            overall_score
        ),

        "comment": _generate_risk_comment(
            score=overall_score,
            ocr_risk=ocr_risk,
            validation_risk=validation_risk,
            tampering_risk=tampering_risk,
            face_risk=face_risk,
            face_verified=face_verified
        ),

        "components": {
            "ocr": round(
                ocr_risk,
                2
            ),

            "validation": round(
                validation_risk,
                2
            ),

            "tampering": round(
                tampering_risk,
                2
            ),

            "face": round(
                face_risk,
                2
            )
        },

        "face_verified": face_verified
    }

    # Store the latest verification state so the server knows
    # the current risk is based on an actual live-face result.
    session["face_verification"] = {
        "identity_status": verification.get(
            "identity_status",
            ""
        ),
        "result": verification.get(
            "result",
            ""
        ),
        "similarity": _safe_float(
            verification.get(
                "similarity",
                0
            ),
            0.0
        ),
        "threshold": _safe_float(
            verification.get(
                "threshold",
                0.40
            ),
            0.40
        )
    }

    _print_overall_risk(
        overall_risk
    )

    # The dashboard must use THIS newly calculated object.
    verification["overall_risk"] = (
        overall_risk
    )

    return jsonify(
        verification
    )


# ============================================================
# FORGOT PASSWORD
# ============================================================

@app.route(
    "/forgot-password"
)
def forgot_password():

    return (
        "Forgot Password feature coming soon."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    create_table()

    app.run(
        debug=True
    )

    app.run(
        debug=True
    )   