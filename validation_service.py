# ============================================================
# DOCUMENT VALIDATION SERVICE
# Supports: PAN + AADHAAR + PASSPORT
# Input: OCR result dictionary
# ============================================================

import re
from datetime import datetime


def _clean(value):
    if value is None:
        return ""
    return str(value).strip()


def _valid_date(value):
    value = _clean(value)
    if not value:
        return False

    try:
        date_value = datetime.strptime(value, "%d/%m/%Y")
        return date_value <= datetime.today()
    except ValueError:
        return False


def validate_document(result):
    """
    Validate OCR output for PAN, Aadhaar and Passport.

    IMPORTANT:
    Passport is handled separately so the passport fields are not lost
    inside the generic PAN/Aadhaar validation output.
    """

    doc_type = _clean(result.get("document_type")).upper()

    # ============================================================
    # PASSPORT
    # ============================================================

    if doc_type == "PASSPORT":

        passport_number = _clean(result.get("passport_number")).upper()
        surname = _clean(result.get("surname")).upper()
        given_name = _clean(result.get("given_name")).upper()
        nationality = _clean(result.get("nationality")).upper()
        dob = _clean(result.get("dob"))
        sex = _clean(result.get("sex") or result.get("gender")).upper()
        place_of_birth = _clean(result.get("place_of_birth")).upper()
        place_of_issue = _clean(result.get("place_of_issue")).upper()
        date_of_issue = _clean(result.get("date_of_issue"))
        date_of_expiry = _clean(result.get("date_of_expiry"))

        checks = {}

        # Passport number
        if passport_number and re.fullmatch(
            r"[A-Z0-9]{6,9}",
            passport_number
        ):
            checks["Passport Number"] = "PASS"
        else:
            checks["Passport Number"] = "FAIL - Missing or invalid"

        # Surname
        if surname and re.fullmatch(r"[A-Z .'-]+", surname):
            checks["Surname"] = "PASS"
        else:
            checks["Surname"] = "FAIL - Missing"

        # Given name
        if given_name and re.fullmatch(r"[A-Z .'-]+", given_name):
            checks["Given Name"] = "PASS"
        else:
            checks["Given Name"] = "FAIL - Missing"

        # Nationality
        if nationality:
            checks["Nationality"] = "PASS"
        else:
            checks["Nationality"] = "FAIL - Missing"

        # DOB
        if _valid_date(dob):
            checks["Date of Birth"] = "PASS"
        else:
            checks["Date of Birth"] = "FAIL - Missing or invalid"

        # Sex
        if sex in ["MALE", "FEMALE"]:
            checks["Sex"] = "PASS"
        else:
            checks["Sex"] = "FAIL - Missing or invalid"

        # Place of birth
        if place_of_birth:
            checks["Place of Birth"] = "PASS"
        else:
            checks["Place of Birth"] = "FAIL - Missing"

        # Place of issue
        if place_of_issue:
            checks["Place of Issue"] = "PASS"
        else:
            checks["Place of Issue"] = "FAIL - Missing"

        # Date of issue
        if _valid_date(date_of_issue):
            checks["Date of Issue"] = "PASS"
        else:
            checks["Date of Issue"] = "FAIL - Missing or invalid"

        # Date of expiry
        expiry_valid = False
        if date_of_expiry:
            try:
                expiry_date = datetime.strptime(
                    date_of_expiry,
                    "%d/%m/%Y"
                )
                expiry_valid = True
            except ValueError:
                expiry_valid = False

        if expiry_valid:
            checks["Date of Expiry"] = "PASS"
        else:
            checks["Date of Expiry"] = "FAIL - Missing or invalid"

        failed = any(status != "PASS" for status in checks.values())
        final_status = "FAIL" if failed else "PASS"

        # ========================================================
        # THIS IS THE PASSPORT TERMINAL OUTPUT
        # ========================================================

        print()
        print("=" * 55)
        print("              PASSPORT INFORMATION")
        print("=" * 55)
        print(f"Document Type  : PASSPORT")
        print(f"Passport No.   : {passport_number or 'Not detected'}")
        print(f"Surname        : {surname or 'Not detected'}")
        print(f"Given Name     : {given_name or 'Not detected'}")
        print(f"Nationality    : {nationality or 'Not detected'}")
        print(f"Date of Birth  : {dob or 'Not detected'}")
        print(f"Sex            : {sex or 'Not detected'}")
        print(
            f"Place of Birth : "
            f"{place_of_birth or 'Not detected'}"
        )
        print(
            f"Place of Issue : "
            f"{place_of_issue or 'Not detected'}"
        )
        print(
            f"Date of Issue  : "
            f"{date_of_issue or 'Not detected'}"
        )
        print(
            f"Date of Expiry : "
            f"{date_of_expiry or 'Not detected'}"
        )
        print("=" * 55)

        print()
        print("=" * 55)
        print("                VALIDATION RESULT")
        print("=" * 55)

        for field, status in checks.items():
            if status == "PASS":
                print(f"{field:<22} : ✓ PASS")
            else:
                print(f"{field:<22} : ✗ {status}")

        print("=" * 55)

        if final_status == "PASS":
            print("FINAL STATUS : ✓ BASIC VALIDATION PASSED")
        else:
            print("FINAL STATUS : ✗ VALIDATION FAILED")

        print("=" * 55)

        return {
            "checks": checks,
            "final_status": final_status
        }

    # ============================================================
    # PAN + AADHAAR
    # ============================================================

    name = _clean(result.get("name"))
    dob = _clean(result.get("dob"))
    gender = _clean(
        result.get("gender") or result.get("sex")
    ).upper()

    pan_number = _clean(
        result.get("pan_number")
    ).upper()

    aadhaar_number = _clean(
        result.get("aadhaar_number")
    )

    checks = {}

    if doc_type in ["PAN", "AADHAAR"]:
        checks["Document Type"] = "PASS"
    else:
        checks["Document Type"] = "FAIL - Unsupported document"

    if not name:
        checks["Name"] = "FAIL - Missing"
    elif not re.fullmatch(r"[A-Za-z .'-]+", name):
        checks["Name"] = "FAIL - Invalid characters"
    else:
        checks["Name"] = "PASS"

    if not dob:
        checks["DOB"] = "FAIL - Missing"
    elif _valid_date(dob):
        checks["DOB"] = "PASS"
    else:
        checks["DOB"] = "FAIL - Invalid date"

    if not gender:
        checks["Gender"] = "FAIL - Missing"
    elif gender in ["MALE", "FEMALE", "TRANSGENDER"]:
        checks["Gender"] = "PASS"
    else:
        checks["Gender"] = "FAIL - Invalid value"

    if doc_type == "PAN":
        if not pan_number:
            checks["PAN Number"] = "FAIL - Missing"
        elif re.fullmatch(
            r"[A-Z]{5}[0-9]{4}[A-Z]",
            pan_number
        ):
            checks["PAN Number"] = "PASS"
        else:
            checks["PAN Number"] = "FAIL - Invalid format"

    elif doc_type == "AADHAAR":
        aadhaar_digits = re.sub(
            r"\s+",
            "",
            aadhaar_number
        )

        if not aadhaar_digits:
            checks["Aadhaar Number"] = "FAIL - Missing"
        elif re.fullmatch(
            r"\d{12}",
            aadhaar_digits
        ):
            checks["Aadhaar Number"] = "PASS"
        else:
            checks["Aadhaar Number"] = "FAIL - Invalid format"

    failed = any(status != "PASS" for status in checks.values())
    final_status = "FAIL" if failed else "PASS"

    # ============================================================
    # PAN OUTPUT
    # ============================================================

    if doc_type == "PAN":
        print()
        print("=" * 40)
        print("        INCOME TAX DEPARTMENT")
        print("             GOVT OF INDIA")
        print("=" * 40)
        print("         DOCUMENT INFORMATION")
        print("=" * 40)
        print(f"Name          : {name or 'Not detected'}")
        print(f"DOB           : {dob or 'Not detected'}")
        print(f"Gender        : {gender or 'Not detected'}")
        print(f"PAN Number    : {pan_number or 'Not detected'}")
        print("=" * 40)

    # ============================================================
    # AADHAAR OUTPUT
    # ============================================================

    elif doc_type == "AADHAAR":
        print()
        print("=" * 40)
        print("         GOVERNMENT OF INDIA")
        print("             AADHAAR")
        print("=" * 40)
        print("         DOCUMENT INFORMATION")
        print("=" * 40)
        print(f"Name          : {name or 'Not detected'}")
        print(f"DOB           : {dob or 'Not detected'}")
        print(f"Gender        : {gender or 'Not detected'}")
        print(
            f"Aadhaar No.   : "
            f"{aadhaar_number or 'Not detected'}"
        )
        print("=" * 40)

    print()
    print("=" * 55)
    print("                VALIDATION RESULT")
    print("=" * 55)

    for field, status in checks.items():
        if status == "PASS":
            print(f"{field:<22} : ✓ PASS")
        else:
            print(f"{field:<22} : ✗ {status}")

    print("=" * 55)

    if final_status == "PASS":
        print("FINAL STATUS : ✓ BASIC VALIDATION PASSED")
    else:
        print("FINAL STATUS : ✗ VALIDATION FAILED")

    print("=" * 55)

    return {
        "checks": checks,
        "final_status": final_status
    }
