# ============================================================
# OVERALL RISK SCORING SERVICE
#
# Combines:
#   OCR
#   Validation
#   Tampering Detection
#   Face Detection / Verification
#
# Output:
#   Fake Risk Percentage
#   Risk Level
#   Comment
# ============================================================


def calculate_ocr_risk(result):
    """
    Calculate OCR-related risk.

    Lower risk = OCR successfully extracted expected fields.
    Higher risk = important fields are missing.
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
            result.get("sex"),
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
        1 for field in fields
        if field is not None
        and str(field).strip() != ""
    )

    completeness = detected / total

    # Missing OCR fields create risk.
    risk = (1.0 - completeness) * 100.0

    return round(risk, 2)


def calculate_validation_risk(validation):
    """
    Convert validation results into a 0-100 risk score.
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
        if str(status).upper() != "PASS"
    )

    risk = (failed / total) * 100.0

    return round(risk, 2)


def calculate_tampering_risk(tampering):
    """
    Tampering service already produces a risk_score.
    """

    if not tampering:
        return 50.0

    if tampering.get("status") == "ERROR":
        return 60.0

    try:
        risk = float(
            tampering.get("risk_score", 0)
        )
    except (TypeError, ValueError):
        risk = 50.0

    return max(0.0, min(100.0, risk))


def calculate_face_risk(
    face_result=None,
    face_verification=None
):
    """
    Face risk.

    Before live verification:
        Face detected     -> low risk
        Face not detected -> high risk

    After live verification:
        VERIFIED          -> 0 risk
        NO MATCH          -> 100 risk
    """

    # --------------------------------------------------------
    # LIVE VERIFICATION RESULT EXISTS
    # --------------------------------------------------------

    if face_verification:

        identity_status = str(
            face_verification.get(
                "identity_status",
                ""
            )
        ).upper().strip()

        result = str(
            face_verification.get(
                "result",
                ""
            )
        ).upper().strip()

        if identity_status == "VERIFIED":
            return 0.0

        if result in [
            "NO MATCH",
            "NOT MATCHED",
            "MISMATCH"
        ]:
            return 100.0

        if identity_status in [
            "SUSPICIOUS",
            "UNVERIFIED",
            "FAILED"
        ]:
            return 80.0

        # If similarity is available,
        # convert it into a risk value.
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

                ratio = similarity / threshold

                if ratio >= 1:
                    return 0.0

                risk = (1.0 - ratio) * 100.0

                return round(
                    max(0.0, min(100.0, risk)),
                    2
                )

        except (
            TypeError,
            ValueError,
            ZeroDivisionError
        ):
            pass

        return 60.0

    # --------------------------------------------------------
    # DOCUMENT FACE DETECTION ONLY
    # --------------------------------------------------------

    if not face_result:
        return 50.0

    if face_result.get("success"):
        return 5.0

    return 80.0


def get_risk_level(score):

    if score <= 20:
        return "LOW RISK"

    if score <= 50:
        return "MEDIUM RISK"

    if score <= 75:
        return "HIGH RISK"

    return "VERY HIGH RISK"


def generate_comment(
    score,
    ocr_risk,
    validation_risk,
    tampering_risk,
    face_risk,
    face_verified=False
):
    """
    Generate an explanation based on the strongest
    risk indicators.
    """

    reasons = []

    if ocr_risk >= 50:
        reasons.append(
            "OCR extraction detected missing or incomplete document information"
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

    if face_verified:
        pass

    elif face_risk >= 75:
        reasons.append(
            "the document face could not be successfully verified"
        )

    elif face_risk >= 30:
        reasons.append(
            "face verification is incomplete or requires attention"
        )

    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    if score <= 20:

        return (
            "Document appears genuine with no significant "
            "inconsistencies found."
        )

    # --------------------------------------------------------
    # MEDIUM RISK
    # --------------------------------------------------------

    if score <= 50:

        if reasons:

            return (
                "Document shows some suspicious indicators. "
                + "; ".join(reasons)
                + ". Further verification is recommended."
            )

        return (
            "Document contains some inconsistencies. "
            "Further verification is recommended."
        )

    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VERY HIGH RISK
    # --------------------------------------------------------

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
    Main overall risk calculation.

    Weights:
        OCR        = 20%
        Validation = 25%
        Tampering  = 30%
        Face       = 25%
    """

    ocr_risk = calculate_ocr_risk(
        result
    )

    validation_risk = calculate_validation_risk(
        validation
    )

    tampering_risk = calculate_tampering_risk(
        tampering
    )

    face_risk = calculate_face_risk(
        face_result,
        face_verification
    )

    # --------------------------------------------------------
    # WEIGHTED SCORE
    # --------------------------------------------------------

    overall_score = (
        (ocr_risk * 0.20)
        +
        (validation_risk * 0.25)
        +
        (tampering_risk * 0.30)
        +
        (face_risk * 0.25)
    )

    overall_score = round(
        max(
            0.0,
            min(100.0, overall_score)
        ),
        2
    )

    risk_level = get_risk_level(
        overall_score
    )

    face_verified = False

    if face_verification:

        face_verified = (
            str(
                face_verification.get(
                    "identity_status",
                    ""
                )
            ).upper()
            == "VERIFIED"
        )

    comment = generate_comment(
        overall_score,
        ocr_risk,
        validation_risk,
        tampering_risk,
        face_risk,
        face_verified
    )

    return {
        "risk_score": overall_score,
        "risk_level": risk_level,
        "comment": comment,

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