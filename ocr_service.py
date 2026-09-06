# ============================================================
# OCR SERVICE
# Supports: PAN, AADHAAR, PASSPORT
# Main function used by Flask:
#     process_document(filepath)
# ============================================================

import re
import cv2
import easyocr
from datetime import datetime


# ============================================================
# EASY OCR
# ============================================================

reader = easyocr.Reader(["en"], gpu=False)


# ============================================================
# COMMON HELPERS
# ============================================================

def load_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Unable to read image: {filepath}")
    return image


def run_ocr(image):
    return reader.readtext(
        image,
        detail=1,
        paragraph=False
    )


def collect_text(results):
    text = []
    for item in results:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            value = str(item[1]).strip()
            if value:
                text.append(value)
    return text


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip())


def clean_alnum(text):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )


def find_date(text):
    if not text:
        return None

    patterns = [
        r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
        r"\b\d{2}[/-]\d{2}[/-]\d{2}\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, str(text))
        if not match:
            continue

        value = match.group().replace("-", "/")
        parts = value.split("/")

        if len(parts) == 3 and len(parts[2]) == 2:
            yy = int(parts[2])
            year = 1900 + yy if yy >= 30 else 2000 + yy
            value = f"{parts[0]}/{parts[1]}/{year}"

        try:
            datetime.strptime(value, "%d/%m/%Y")
            return value
        except ValueError:
            pass

    return None


def find_date_after_label(texts, labels, max_next=5):
    for i, text in enumerate(texts):
        lower = text.lower()

        if not any(label.lower() in lower for label in labels):
            continue

        value = find_date(text)
        if value:
            return value

        for j in range(i + 1, min(i + 1 + max_next, len(texts))):
            value = find_date(texts[j])
            if value:
                return value

    return None


def find_value_after_label(texts, labels, max_next=5):
    """
    Generic text-list fallback.

    This is intentionally conservative because passport fields are handled
    more accurately by the bounding-box based functions below.
    """
    normalized_labels = [re.sub(r"[^a-z]", "", x.lower()) for x in labels]

    for i, text in enumerate(texts):
        raw = normalize_text(text)
        lower_clean = re.sub(r"[^a-z]", "", raw.lower())

        matched_label = None
        for label in normalized_labels:
            if label and label in lower_clean:
                matched_label = label
                break

        if not matched_label:
            continue

        # Value on the same OCR line.
        lower_raw = raw.lower()
        for original_label in labels:
            pos = lower_raw.find(original_label.lower())
            if pos >= 0:
                value = raw[pos + len(original_label):].strip(" :-")
                if value:
                    return normalize_text(value)

        if ":" in raw:
            value = raw.split(":", 1)[1].strip()
            if value:
                return normalize_text(value)

        # Value on the next OCR lines.
        for j in range(i + 1, min(i + 1 + max_next, len(texts))):
            value = normalize_text(texts[j])
            if not value:
                continue

            compact = re.sub(r"[^a-z]", "", value.lower())
            if any(label in compact for label in normalized_labels):
                continue
            if find_date(value):
                continue

            return value

    return None


def _ocr_item_data(item):
    """Return (bbox, text, confidence) for an EasyOCR detection."""
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None

    bbox = item[0]
    text = normalize_text(item[1])
    try:
        confidence = float(item[2]) if len(item) >= 3 else 0.0
    except Exception:
        confidence = 0.0

    if not text:
        return None

    try:
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
    except Exception:
        return None

    return {
        "bbox": bbox,
        "text": text,
        "confidence": confidence,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "cx": center_x,
        "cy": center_y,
        "width": max(1.0, right - left),
        "height": max(1.0, bottom - top),
    }


def _compact_text(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _label_match_score(text, labels):
    compact = _compact_text(text)
    best = 0

    for label in labels:
        lc = _compact_text(label)
        if not lc:
            continue
        if compact == lc:
            best = max(best, 100)
        elif lc in compact:
            best = max(best, 90)
    return best


def _is_bad_passport_value(text):
    """
    Reject things that are almost certainly labels, MRZ data, or unrelated
    passport boilerplate.
    """
    value = normalize_text(text)
    compact = _compact_text(value)

    if not value:
        return True

    bad_words = [
        "placeofbirth",
        "placeofissue",
        "dateofbirth",
        "dateofissue",
        "dateofexpiry",
        "surname",
        "givenname",
        "nationality",
        "passport",
        "republic",
        "india",
        "government",
        "republicofindia",
        "passportnumber",
        "sex",
        "name",
        "issue",
        "birth",
        "expiry",
        "expiration",
        "date",
        "cfissue",
    ]

    if any(word in compact for word in bad_words):
        return True

    # MRZ lines should never be used as normal-field values.
    if "<" in value:
        return True

    if len(value) > 70:
        return True

    return False


def _clean_passport_place(value):
    if not value:
        return None

    value = normalize_text(value)
    value = value.strip(" :-.,")
    value = re.sub(r"^[|Iil:;,_-]+", "", value)
    value = re.sub(r"[|;,_-]+$", "", value)
    value = normalize_text(value)

    if _is_bad_passport_value(value):
        return None

    # Passport places on the bio-data page are alphabetic location names.
    # Reject OCR artefacts containing digits, slash-heavy date strings,
    # or mixed alphanumeric strings such as "23/0312036E8".
    if re.search(r"\d", value):
        return None

    # Keep useful punctuation such as commas in locations.
    value = re.sub(r"[^A-Za-z ,.'()/&-]", "", value)
    value = normalize_text(value).upper()

    if len(value) < 2:
        return None

    return value


def _extract_inline_value(text, labels):
    """Extract a value when EasyOCR puts label and value in one box."""
    raw = normalize_text(text)
    lower = raw.lower()

    # Longest labels first.
    for label in sorted(labels, key=len, reverse=True):
        pos = lower.find(label.lower())
        if pos < 0:
            continue

        value = raw[pos + len(label):]
        value = value.lstrip(" :-|")
        value = normalize_text(value)

        if value:
            return value

    compact = _compact_text(raw)
    for label in sorted(labels, key=len, reverse=True):
        lc = _compact_text(label)
        if lc and lc in compact:
            # Recover approximately from the original text after the label.
            pos = lower.find(label.lower().replace(" ", ""))
            if pos >= 0:
                value = raw[pos + len(label):].lstrip(" :-|")
                if value:
                    return normalize_text(value)

    return None


def _passport_items_from_passes(passes):
    """
    Preserve EasyOCR bounding boxes across every passport OCR pass.
    Duplicate text is retained when it comes from a better-confidence box.
    """
    items = []

    for result in passes:
        for raw_item in result:
            data = _ocr_item_data(raw_item)
            if data:
                items.append(data)

    return items


def _find_passport_spatial_value(items, labels, value_type="place"):
    """
    Locate a passport field using OCR geometry.

    Passport labels and their values are commonly either:
      LABEL : VALUE
    or
      LABEL
      VALUE

    We search both to the right and below the label, across all OCR passes.
    """
    label_items = []
    for item in items:
        score = _label_match_score(item["text"], labels)
        if score:
            item = dict(item)
            item["label_score"] = score
            label_items.append(item)

    if not label_items:
        return None

    candidates = []

    for label_item in label_items:
        inline = _extract_inline_value(label_item["text"], labels)

        if inline:
            if value_type == "date":
                date = find_date(inline)
                if date:
                    candidates.append((1200 + label_item["confidence"] * 10, date))
            else:
                place = _clean_passport_place(inline)
                if place:
                    candidates.append(
                        (1200 + label_item["confidence"] * 10, place)
                    )

        label_height = label_item["height"]

        for item in items:
            if item is label_item:
                continue

            text = item["text"]
            if _is_bad_passport_value(text):
                continue

            # Never treat another label as a field value.
            if _label_match_score(text, labels):
                continue

            if value_type == "date":
                candidate_value = find_date(text)
                if not candidate_value:
                    continue
            else:
                candidate_value = _clean_passport_place(text)
                if not candidate_value:
                    continue
                if find_date(candidate_value):
                    continue

            dx = item["left"] - label_item["right"]
            vertical_gap = item["top"] - label_item["bottom"]
            same_row = (
                abs(item["cy"] - label_item["cy"])
                <= max(label_height * 1.8, item["height"] * 1.5)
            )

            # Candidate directly to the right of the label.
            if dx >= -5 and dx <= max(900, label_item["width"] * 8) and same_row:
                distance = max(0.0, dx) + abs(item["cy"] - label_item["cy"]) * 1.5
                score = (
                    900
                    + label_item["label_score"]
                    + item["confidence"] * 20
                    - distance
                )
                candidates.append((score, candidate_value))

            # Candidate below the label.
            below = vertical_gap >= -5
            aligned = (
                item["right"] >= label_item["left"] - label_item["width"] * 1.5
                and item["left"] <= label_item["right"] + label_item["width"] * 1.5
            )

            if below and aligned and vertical_gap <= max(500, label_height * 8):
                distance = vertical_gap + abs(item["cx"] - label_item["cx"]) * 0.8
                score = (
                    700
                    + label_item["label_score"]
                    + item["confidence"] * 20
                    - distance
                )
                candidates.append((score, candidate_value))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _find_passport_date_by_label(items, labels):
    return _find_passport_spatial_value(
        items,
        labels,
        value_type="date"
    )


def _find_passport_place_by_label(items, labels):
    return _find_passport_spatial_value(
        items,
        labels,
        value_type="place"
    )


# ============================================================
# DOCUMENT DETECTION
# ============================================================

def looks_like_passport_mrz(texts):
    for text in texts:
        cleaned = clean_alnum(text).replace("P", "P", 1)

        raw = str(text).upper().replace(" ", "")
        raw = re.sub(r"[^A-Z0-9<]", "", raw)

        if raw.startswith("P<") and len(raw) >= 15:
            return True

        if (
            len(raw) >= 30
            and "<" in raw
            and re.search(r"\d", raw)
        ):
            return True

    return False


def detect_document_type(texts):
    if not texts:
        return "UNKNOWN"

    upper = " ".join(texts).upper()

    # Passport MUST be checked first.
    passport_words = [
        "PASSPORT",
        "SURNAME",
        "GIVEN NAME",
        "GIVENNAME",
        "NATIONALITY",
        "PLACE OF BIRTH",
        "PLACEOFBIRTH",
        "PLACE OF ISSUE",
        "PLACEOFISSUE",
        "DATE OF ISSUE",
        "DATEOFISSUE"
    ]

    passport_score = sum(
        1 for word in passport_words
        if word in upper
    )

    if looks_like_passport_mrz(texts):
        passport_score += 5

    if passport_score >= 2:
        return "PASSPORT"

    # PAN
    for text in texts:
        cleaned = clean_alnum(text)
        if re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", cleaned):
            return "PAN"

    if (
        "INCOME TAX" in upper
        or "INCOMETAX" in upper
        or "PERMANENT ACCOUNT NUMBER" in upper
        or "PERMANENTACCOUNTNUMBER" in upper
    ):
        return "PAN"

    # Aadhaar
    if (
        "AADHAAR" in upper
        or "UIDAI" in upper
        or "UNIQUE IDENTIFICATION" in upper
        or "UNIQUEIDENTIFICATION" in upper
    ):
        return "AADHAAR"

    for text in texts:
        if re.search(
            r"\b\d{4}\s?\d{4}\s?\d{4}\b",
            text
        ):
            return "AADHAAR"

    return "UNKNOWN"


# ============================================================
# GENDER
# ============================================================

def extract_gender(texts):
    for text in texts:
        cleaned = re.sub(
            r"[^A-Z]",
            "",
            str(text).upper()
        )

        if cleaned == "MALE":
            return "MALE"

        if cleaned == "FEMALE":
            return "FEMALE"

    return None


# ============================================================
# NAME CLEANING
# ============================================================

def clean_person_name(value):
    if not value:
        return None

    value = re.sub(
        r"[^A-Za-z ]",
        "",
        str(value)
    )

    value = normalize_text(value).upper()

    if not value:
        return None

    return value


# ============================================================
# PAN
# ============================================================

def extract_pan(texts):
    pan_number = None

    for text in texts:
        cleaned = clean_alnum(text)

        match = re.search(
            r"[A-Z]{5}[0-9]{4}[A-Z]",
            cleaned
        )

        if match:
            pan_number = match.group()
            break

    dob = find_date_after_label(
        texts,
        ["date of birth", "dob", "birth"]
    )

    if not dob:
        for text in texts:
            dob = find_date(text)
            if dob:
                break

    gender = extract_gender(texts)

    name = None

    # Prefer text following a Name label.
    for i, text in enumerate(texts):
        if "name" not in text.lower():
            continue

        if "father" in text.lower():
            continue

        if ":" in text:
            candidate = text.split(":", 1)[1]
            name = clean_person_name(candidate)

        if not name:
            for j in range(
                i + 1,
                min(i + 5, len(texts))
            ):
                candidate = clean_person_name(texts[j])
                if (
                    candidate
                    and len(candidate) >= 3
                    and not find_date(candidate)
                    and "father" not in candidate.lower()
                ):
                    name = candidate
                    break

        if name:
            break

    result = {
        "document_type": "PAN",
        "name": name,
        "dob": dob,
        "gender": gender,
        "sex": gender,
        "pan_number": pan_number,
        "aadhaar_number": None,
        "passport_number": None,
        "surname": None,
        "given_name": None,
        "nationality": None,
        "place_of_birth": None,
        "place_of_issue": None,
        "date_of_issue": None,
        "date_of_expiry": None,
        "mrz_lines": [],
        "face_detected": False
    }

    print()
    print("=" * 40)
    print("        INCOME TAX DEPARTMENT")
    print("             GOVT OF INDIA")
    print("=" * 40)
    print("         DOCUMENT INFORMATION")
    print("=" * 40)
    print(f"Document Type : PAN")
    print(f"Name          : {name or 'Not detected'}")
    print(f"DOB           : {dob or 'Not detected'}")
    print(f"Gender        : {gender or 'Not detected'}")
    print(f"PAN Number    : {pan_number or 'Not detected'}")
    print("=" * 40)

    return result


# ============================================================
# AADHAAR
# ============================================================

def extract_aadhaar(texts):
    combined = " ".join(texts)

    aadhaar_number = None

    match = re.search(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        combined
    )

    if match:
        digits = re.sub(r"\D", "", match.group())
        aadhaar_number = (
            f"{digits[:4]} {digits[4:8]} {digits[8:12]}"
            if len(digits) == 12
            else digits
        )

    dob = find_date_after_label(
        texts,
        ["date of birth", "dob", "birth", "year of birth"]
    )

    if not dob:
        for text in texts:
            dob = find_date(text)
            if dob:
                break

    gender = extract_gender(texts)

    name = None

    for i, text in enumerate(texts):
        if "name" not in text.lower():
            continue

        if ":" in text:
            name = clean_person_name(
                text.split(":", 1)[1]
            )

        if not name:
            for j in range(
                i + 1,
                min(i + 5, len(texts))
            ):
                candidate = clean_person_name(texts[j])

                if (
                    candidate
                    and len(candidate) >= 3
                    and "government" not in candidate.lower()
                    and "india" not in candidate.lower()
                    and not find_date(candidate)
                ):
                    name = candidate
                    break

        if name:
            break

    result = {
        "document_type": "AADHAAR",
        "name": name,
        "dob": dob,
        "gender": gender,
        "sex": gender,
        "aadhaar_number": aadhaar_number,
        "pan_number": None,
        "passport_number": None,
        "surname": None,
        "given_name": None,
        "nationality": None,
        "place_of_birth": None,
        "place_of_issue": None,
        "date_of_issue": None,
        "date_of_expiry": None,
        "mrz_lines": [],
        "face_detected": False
    }

    print()
    print("=" * 40)
    print("         GOVERNMENT OF INDIA")
    print("             AADHAAR")
    print("=" * 40)
    print("         DOCUMENT INFORMATION")
    print("=" * 40)
    print(f"Document Type : AADHAAR")
    print(f"Name          : {name or 'Not detected'}")
    print(f"DOB           : {dob or 'Not detected'}")
    print(f"Gender        : {gender or 'Not detected'}")
    print(f"Aadhaar No.   : {aadhaar_number or 'Not detected'}")
    print("=" * 40)

    return result


# ============================================================
# PASSPORT MRZ
# ============================================================

def clean_mrz(text):
    return re.sub(
        r"[^A-Z0-9<]",
        "",
        str(text).upper().replace(" ", "")
    )


def mrz_score(text):
    text = clean_mrz(text)

    score = 0

    if text.startswith("P<"):
        score += 100

    if len(text) >= 30:
        score += 20

    if "<" in text:
        score += min(text.count("<") * 2, 20)

    if re.search(r"\d", text):
        score += 5

    if 38 <= len(text) <= 50:
        score += 15

    return score


def find_passport_mrz(results):
    candidates = []

    for item in results:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[1])
            confidence = (
                float(item[2])
                if len(item) >= 3
                else 0
            )
        else:
            text = str(item)
            confidence = 0

        cleaned = clean_mrz(text)

        if not cleaned:
            continue

        if (
            cleaned.startswith("P<")
            or len(cleaned) >= 30
            or ("<" in cleaned and re.search(r"\d", cleaned))
        ):
            candidates.append(
                (cleaned, confidence)
            )

    if not candidates:
        return []

    first_candidates = [
        x for x in candidates
        if x[0].startswith("P<")
    ]

    first = None
    if first_candidates:
        first = max(
            first_candidates,
            key=lambda x: (
                mrz_score(x[0]),
                x[1]
            )
        )

    second_candidates = [
        x for x in candidates
        if not first or x[0] != first[0]
    ]

    second_candidates = [
        x for x in second_candidates
        if len(x[0]) >= 30
        and re.search(r"\d", x[0])
    ]

    second = None
    if second_candidates:
        second = max(
            second_candidates,
            key=lambda x: (
                mrz_score(x[0]),
                x[1]
            )
        )

    if first and second:
        return [
            first[0],
            second[0]
        ]

    # OCR may split the MRZ.
    if first:
        pieces = [
            x[0] for x in candidates
            if x[0] != first[0]
        ]

        joined = first[0]

        for piece in pieces:
            joined += piece

            if len(joined) >= 88:
                return [
                    joined[:44],
                    joined[44:88]
                ]

    return [first[0]] if first else []


def extract_mrz_name(mrz_lines):
    for line in mrz_lines:
        line = clean_mrz(line)

        if not line.startswith("P<"):
            continue

        name_section = line[5:]
        parts = name_section.split("<<", 1)

        if len(parts) != 2:
            continue

        surname = normalize_text(
            parts[0].replace("<", " ")
        ).upper()

        given_name = normalize_text(
            parts[1].replace("<", " ")
        ).upper()

        surname = re.sub(
            r"[^A-Z ]",
            "",
            surname
        ).strip()

        given_name = re.sub(
            r"[^A-Z ]",
            "",
            given_name
        ).strip()

        return (
            surname or None,
            given_name or None
        )

    return None, None


def extract_mrz_data(mrz_lines):
    for line in mrz_lines:
        cleaned = clean_mrz(line)

        if cleaned.startswith("P<"):
            continue

        if len(cleaned) < 27:
            continue

        passport_number = (
            cleaned[0:9]
            .replace("<", "")
        )

        nationality = cleaned[10:13]
        dob_raw = cleaned[13:19]
        sex = cleaned[20:21]
        expiry_raw = cleaned[21:27]

        return {
            "passport_number": passport_number or None,
            "nationality": nationality or None,
            "dob_raw": dob_raw or None,
            "sex": sex or None,
            "expiry_raw": expiry_raw or None
        }

    return {
        "passport_number": None,
        "nationality": None,
        "dob_raw": None,
        "sex": None,
        "expiry_raw": None
    }


def convert_mrz_date(value, date_type):
    if not value or len(value) != 6 or not value.isdigit():
        return None

    yy = int(value[:2])
    mm = int(value[2:4])
    dd = int(value[4:6])

    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None

    if date_type == "dob":
        year = 1900 + yy if yy >= 30 else 2000 + yy
    else:
        year = 2000 + yy

    try:
        datetime(
            year,
            mm,
            dd
        )
    except ValueError:
        return None

    return f"{dd:02d}/{mm:02d}/{year}"


def convert_nationality(code):
    if not code:
        return None

    mapping = {
        "IND": "INDIAN",
        "USA": "AMERICAN",
        "GBR": "BRITISH",
        "CAN": "CANADIAN",
        "AUS": "AUSTRALIAN",
        "FRA": "FRENCH",
        "DEU": "GERMAN",
        "JPN": "JAPANESE",
        "CHN": "CHINESE",
        "SGP": "SINGAPOREAN",
        "ARE": "EMIRATI"
    }

    return mapping.get(
        code.upper(),
        code.upper()
    )


def convert_passport_sex(value):
    if not value:
        return None

    value = value.upper().strip()

    if value == "M":
        return "MALE"

    if value == "F":
        return "FEMALE"

    return None


# ============================================================
# PASSPORT NORMAL FIELDS
# ============================================================


def _passport_fixed_layout_fields(rotated):
    """
    Read the three fields that EasyOCR most often confuses on the
    standard Indian passport bio-data page.

    The image is already rotated upright before this function is called.
    Small row crops stop OCR from borrowing text from the neighbouring row.
    """
    result = {
        "place_of_birth": None,
        "place_of_issue": None,
        "date_of_issue": None,
    }

    try:
        h, w = rotated.shape[:2]

        # Standard passport bio-data block: left/middle section.
        x1 = int(w * 0.23)
        x2 = int(w * 0.66)

        rows = {
            "place_of_birth": (0.31, 0.43),
            "place_of_issue": (0.37, 0.50),
            "date_of_issue": (0.43, 0.57),
        }

        for field, (y1_ratio, y2_ratio) in rows.items():
            y1 = max(0, int(h * y1_ratio))
            y2 = min(h, int(h * y2_ratio))

            crop = rotated[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(
                crop,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC
            )

            ocr_result = reader.readtext(
                crop,
                detail=1,
                paragraph=False,
                text_threshold=0.20,
                low_text=0.05,
                link_threshold=0.10
            )

            row_texts = []

            for item in ocr_result:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                ):
                    value = normalize_text(item[1])
                    if value:
                        row_texts.append(value)

            if not row_texts:
                continue

            combined = " ".join(row_texts)

            if field == "place_of_birth":
                value = _extract_inline_value(
                    combined,
                    [
                        "Place of Birth",
                        "PlaceofBirth",
                    ]
                )

                if not value:
                    value = find_value_after_label(
                        row_texts,
                        [
                            "place of birth",
                            "placeofbirth",
                        ],
                        max_next=3
                    )

                value = _clean_passport_place(value)

                if value and "ISSUE" not in value:
                    result[field] = value

            elif field == "place_of_issue":
                value = _extract_inline_value(
                    combined,
                    [
                        "Place of Issue",
                        "PlaceofIssue",
                    ]
                )

                if not value:
                    value = find_value_after_label(
                        row_texts,
                        [
                            "place of issue",
                            "placeofissue",
                        ],
                        max_next=3
                    )

                value = _clean_passport_place(value)

                if value and "ISSUE" not in value:
                    result[field] = value

            elif field == "date_of_issue":
                value = _extract_inline_value(
                    combined,
                    [
                        "Date of Issue",
                        "DateofIssue",
                    ]
                )

                date_value = (
                    find_date(value)
                    if value
                    else None
                )

                if not date_value:
                    date_value = find_date_after_label(
                        row_texts,
                        [
                            "date of issue",
                            "dateofissue",
                        ],
                        max_next=3
                    )

                if date_value:
                    result[field] = date_value

    except Exception:
        # Fallback only. Main OCR must never fail because of this helper.
        pass

    return result


def extract_passport_normal_fields(passes, texts, rotated=None):
    """
    Extract the passport's normal visual fields.

    The previous implementation flattened OCR into a text list and assumed
    the value would occur within the next few OCR detections. That is not
    reliable for passports because EasyOCR returns detections in spatial
    order, not guaranteed label/value order.

    This version first uses bounding boxes from every OCR pass, then falls
    back to the text-list method.
    """
    fixed = {
        "place_of_birth": None,
        "place_of_issue": None,
        "date_of_issue": None,
    }

    if rotated is not None:
        fixed = _passport_fixed_layout_fields(rotated)

    items = _passport_items_from_passes(passes)

    place_of_birth = _find_passport_place_by_label(
        items,
        [
            "Place of Birth",
            "PlaceofBirth",
            "Place Of Birth",
            "PLACE OF BIRTH",
        ]
    )

    place_of_issue = _find_passport_place_by_label(
        items,
        [
            "Place of Issue",
            "PlaceofIssue",
            "Place Of Issue",
            "PLACE OF ISSUE",
        ]
    )

    date_of_issue = _find_passport_date_by_label(
        items,
        [
            "Date of Issue",
            "DateofIssue",
            "Date Of Issue",
            "DATE OF ISSUE",
            "Date Issue",
            "Issue Date",
        ]
    )

    # Fixed-layout crop is preferred when it found a real value.
    # It is specifically designed to prevent neighbouring passport rows
    # from being mixed together.
    if fixed["place_of_birth"]:
        place_of_birth = fixed["place_of_birth"]

    if fixed["place_of_issue"]:
        place_of_issue = fixed["place_of_issue"]

    if fixed["date_of_issue"]:
        date_of_issue = fixed["date_of_issue"]

    # Text-list fallback.
    if not place_of_birth:
        place_of_birth = find_value_after_label(
            texts,
            [
                "place of birth",
                "placeofbirth",
            ],
            max_next=8,
        )

    if not place_of_issue:
        place_of_issue = find_value_after_label(
            texts,
            [
                "place of issue",
                "placeofissue",
            ],
            max_next=8,
        )

    if not date_of_issue:
        date_of_issue = find_date_after_label(
            texts,
            [
                "date of issue",
                "dateofissue",
                "date issue",
                "issue date",
            ],
            max_next=8,
        )

    place_of_birth = _clean_passport_place(place_of_birth)
    place_of_issue = _clean_passport_place(place_of_issue)

    return {
        "place_of_birth": place_of_birth,
        "place_of_issue": place_of_issue,
        "date_of_issue": date_of_issue,
    }


def find_passport_number(texts):
    for text in texts:
        cleaned = clean_alnum(text)

        # MRZ-style passport number fallback.
        match = re.search(
            r"\b[A-Z]{1,2}[0-9]{6,7}\b",
            cleaned
        )

        if match:
            return match.group()

    return None


# ============================================================
# PASSPORT OCR
# ============================================================

def passport_ocr_passes(rotated):
    passes = []

    # Original rotated image
    try:
        passes.append(
            reader.readtext(
                rotated,
                detail=1,
                paragraph=False,
                text_threshold=0.30,
                low_text=0.15,
                link_threshold=0.15
            )
        )
    except Exception:
        pass

    # Enlarged image
    try:
        enlarged = cv2.resize(
            rotated,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC
        )

        passes.append(
            reader.readtext(
                enlarged,
                detail=1,
                paragraph=False,
                text_threshold=0.25,
                low_text=0.10,
                link_threshold=0.10
            )
        )
    except Exception:
        pass

    # Grayscale + threshold
    try:
        gray = cv2.cvtColor(
            rotated,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        passes.append(
            reader.readtext(
                gray,
                detail=1,
                paragraph=False,
                text_threshold=0.25,
                low_text=0.10,
                link_threshold=0.10
            )
        )

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        passes.append(
            reader.readtext(
                binary,
                detail=1,
                paragraph=False,
                text_threshold=0.20,
                low_text=0.05,
                link_threshold=0.10
            )
        )

        # Adaptive threshold for small/light passport field text.
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        passes.append(
            reader.readtext(
                adaptive,
                detail=1,
                paragraph=False,
                text_threshold=0.20,
                low_text=0.05,
                link_threshold=0.10
            )
        )

        # Mild sharpening without destroying the original geometry.
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        sharpened = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

        passes.append(
            reader.readtext(
                sharpened,
                detail=1,
                paragraph=False,
                text_threshold=0.20,
                low_text=0.05,
                link_threshold=0.10
            )
        )

    except Exception:
        pass

    return [
        result
        for result in passes
        if result
    ]


def flatten_passport_results(passes):
    texts = []
    seen = set()

    for result in passes:
        for item in result:
            if len(item) < 2:
                continue

            text = str(item[1]).strip()

            if not text:
                continue

            key = text.upper()

            if key in seen:
                continue

            seen.add(key)
            texts.append(text)

    return texts


def best_passport_mrz(passes):
    best = []

    for result in passes:
        lines = find_passport_mrz(result)

        if len(lines) >= 2:
            score = (
                mrz_score(lines[0])
                + mrz_score(lines[1])
            )

            if not best or score > best[0]:
                best = (
                    score,
                    lines
                )

    return best[1] if best else []


def extract_passport(image):
    rotated = cv2.rotate(
        image,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )

    passes = passport_ocr_passes(
        rotated
    )

    if not passes:
        raise ValueError(
            "Passport OCR failed."
        )

    texts = flatten_passport_results(
        passes
    )

    mrz_lines = best_passport_mrz(
        passes
    )

    surname, given_name = extract_mrz_name(
        mrz_lines
    )

    mrz = extract_mrz_data(
        mrz_lines
    )

    # MRZ is the most reliable source for these fields.
    passport_number = (
        mrz["passport_number"]
        or find_passport_number(texts)
    )

    nationality = convert_nationality(
        mrz["nationality"]
    )

    dob = convert_mrz_date(
        mrz["dob_raw"],
        "dob"
    )

    sex = convert_passport_sex(
        mrz["sex"]
    )

    # Expiry comes directly from the MRZ and is therefore kept
    # independent from the visually detected Date of Issue field.
    expiry = convert_mrz_date(
        mrz["expiry_raw"],
        "expiry"
    )

    normal = extract_passport_normal_fields(
        passes,
        texts,
        rotated
    )

    # --------------------------------------------------------
    # DOB fallback
    # --------------------------------------------------------
    if not dob:
        dob = find_date_after_label(
            texts,
            [
                "date of birth",
                "dateofbirth",
                "birth"
            ],
            max_next=8
        )

    # --------------------------------------------------------
    # EXPIRY FALLBACK
    # --------------------------------------------------------
    if not expiry:
        items = _passport_items_from_passes(passes)
        expiry = _find_passport_date_by_label(
            items,
            [
                "date of expiry",
                "dateofexpiry",
                "expiry",
                "expiration",
            ],
        )

    if not expiry:
        expiry = find_date_after_label(
            texts,
            [
                "date of expiry",
                "dateofexpiry",
                "expiry",
                "expiration"
            ],
            max_next=8
        )

    # --------------------------------------------------------
    # DATE OF ISSUE
    # --------------------------------------------------------
    # IMPORTANT: do not trust a spatial candidate blindly. EasyOCR can
    # associate the Date of Issue label with the neighbouring expiry row.
    date_of_issue = normal.get("date_of_issue")

    # Collect every valid date detected by OCR.
    date_candidates = []
    seen_dates = set()

    for text in texts:
        value = find_date(text)
        if not value or value in seen_dates:
            continue
        seen_dates.add(value)
        date_candidates.append(value)

    # Also inspect every OCR box because a date may be embedded in a
    # detection that was not present in the flattened text list.
    items = _passport_items_from_passes(passes)
    for item in items:
        value = find_date(item.get("text", ""))
        if not value or value in seen_dates:
            continue
        seen_dates.add(value)
        date_candidates.append(value)

    # If expiry is known, the issue date must be an earlier date.
    # For the supplied passport this gives 24/03/2026 rather than
    # incorrectly reusing 23/03/2036.
    if expiry:
        try:
            expiry_date = datetime.strptime(
                expiry,
                "%d/%m/%Y"
            )

            valid_issue_candidates = []

            for value in date_candidates:
                try:
                    candidate_date = datetime.strptime(
                        value,
                        "%d/%m/%Y"
                    )
                except ValueError:
                    continue

                if candidate_date < expiry_date:
                    valid_issue_candidates.append(candidate_date)

            # Remove DOB from consideration when we already know it.
            if dob:
                try:
                    dob_date = datetime.strptime(
                        dob,
                        "%d/%m/%Y"
                    )
                    valid_issue_candidates = [
                        d
                        for d in valid_issue_candidates
                        if d != dob_date
                    ]
                except ValueError:
                    pass

            if valid_issue_candidates:
                best_issue = max(valid_issue_candidates)
                date_of_issue = best_issue.strftime(
                    "%d/%m/%Y"
                )

        except ValueError:
            pass

    # If expiry was not available, retain the spatially detected issue date.
    if not date_of_issue:
        date_of_issue = find_date_after_label(
            texts,
            [
                "date of issue",
                "dateofissue",
                "date issue",
                "issue date"
            ],
            max_next=8
        )

    # Final consistency check.
    if date_of_issue and expiry:
        try:
            issue_date = datetime.strptime(
                date_of_issue,
                "%d/%m/%Y"
            )
            expiry_date = datetime.strptime(
                expiry,
                "%d/%m/%Y"
            )

            if issue_date >= expiry_date:
                date_of_issue = None
        except ValueError:
            date_of_issue = None

    # --------------------------------------------------------
    # SEX FALLBACK
    # --------------------------------------------------------
    if not sex:
        sex = extract_gender(texts)

    # --------------------------------------------------------
    # PLACE SAFETY
    # --------------------------------------------------------
    # Reject neighbouring date/label artefacts.
    for key in ["place_of_birth", "place_of_issue"]:
        value = normal.get(key)
        if value:
            compact = _compact_text(value)

            if (
                re.search(r"\d", value)
                or "issue" in compact
                or "birth" in compact
                or "expiry" in compact
                or "date" in compact
            ):
                normal[key] = None

    # If the spatial result was rejected, retry from the OCR boxes.
    if not normal.get("place_of_birth") or not normal.get("place_of_issue"):
        items = _passport_items_from_passes(passes)

        if not normal.get("place_of_birth"):
            normal["place_of_birth"] = _find_passport_place_by_label(
                items,
                [
                    "Place of Birth",
                    "PlaceofBirth",
                    "Place Of Birth",
                ]
            )

        if not normal.get("place_of_issue"):
            normal["place_of_issue"] = _find_passport_place_by_label(
                items,
                [
                    "Place of Issue",
                    "PlaceofIssue",
                    "Place Of Issue",
                ]
            )

    name = None

    if given_name and surname:
        name = f"{given_name} {surname}"
    elif given_name:
        name = given_name
    elif surname:
        name = surname

    result = {
        "document_type": "PASSPORT",
        "name": name,
        "surname": surname,
        "given_name": given_name,
        "passport_number": passport_number,
        "nationality": nationality,
        "dob": dob,
        "gender": sex,
        "sex": sex,
        "place_of_birth": normal.get("place_of_birth"),
        "place_of_issue": normal.get("place_of_issue"),
        "date_of_issue": date_of_issue,
        "date_of_expiry": expiry,
        "pan_number": None,
        "aadhaar_number": None,
        "mrz_lines": mrz_lines,
        "face_detected": False
    }

    # ========================================================
    # CLEAN PASSPORT TERMINAL OUTPUT
    # ========================================================
    print()
    print("=" * 55)
    print("              PASSPORT INFORMATION")
    print("=" * 55)
    print("Document Type  : PASSPORT")
    print(f"Passport No.   : {passport_number or 'Not detected'}")
    print(f"Surname        : {surname or 'Not detected'}")
    print(f"Given Name     : {given_name or 'Not detected'}")
    print(f"Nationality    : {nationality or 'Not detected'}")
    print(f"Date of Birth  : {dob or 'Not detected'}")
    print(f"Sex            : {sex or 'Not detected'}")
    print(
        f"Place of Birth : "
        f"{normal.get('place_of_birth') or 'Not detected'}"
    )
    print(
        f"Place of Issue : "
        f"{normal.get('place_of_issue') or 'Not detected'}"
    )
    print(
        f"Date of Issue  : "
        f"{date_of_issue or 'Not detected'}"
    )
    print(
        f"Date of Expiry : "
        f"{expiry or 'Not detected'}"
    )
    print("=" * 55)

    return result


# ============================================================
# UNKNOWN
# ============================================================

def unknown_result():
    return {
        "document_type": "UNKNOWN",
        "name": None,
        "dob": None,
        "gender": None,
        "sex": None,
        "pan_number": None,
        "aadhaar_number": None,
        "passport_number": None,
        "surname": None,
        "given_name": None,
        "nationality": None,
        "place_of_birth": None,
        "place_of_issue": None,
        "date_of_issue": None,
        "date_of_expiry": None,
        "mrz_lines": [],
        "face_detected": False
    }


# ============================================================
# MAIN FUNCTION
# ============================================================

def process_document(filepath):
    print()
    print("=" * 60)
    print("             DOCUMENT OCR STARTED")
    print("=" * 60)

    image = load_image(filepath)

    # --------------------------------------------------------
    # First OCR: normal orientation
    # --------------------------------------------------------

    normal_results = run_ocr(image)
    normal_text = collect_text(
        normal_results
    )

    document_type = detect_document_type(
        normal_text
    )

    print(
        f"Detected document: {document_type}"
    )

    # --------------------------------------------------------
    # Direct extraction
    # --------------------------------------------------------

    if document_type == "PAN":
        return extract_pan(normal_text)

    if document_type == "AADHAAR":
        return extract_aadhaar(normal_text)

    if document_type == "PASSPORT":
        return extract_passport(image)

    # --------------------------------------------------------
    # Passport fallback:
    # the passport may be sideways and therefore impossible
    # to identify from the first OCR pass.
    # --------------------------------------------------------

    rotated = cv2.rotate(
        image,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )

    rotated_results = run_ocr(
        rotated
    )

    rotated_text = collect_text(
        rotated_results
    )

    rotated_type = detect_document_type(
        rotated_text
    )

    if (
        rotated_type == "PASSPORT"
        or looks_like_passport_mrz(rotated_text)
    ):
        return extract_passport(image)

    if rotated_type == "PAN":
        return extract_pan(rotated_text)

    if rotated_type == "AADHAAR":
        return extract_aadhaar(rotated_text)

    # --------------------------------------------------------
    # Last chance: MRZ can be present even when keywords
    # were not detected.
    # --------------------------------------------------------

    if find_passport_mrz(rotated_results):
        return extract_passport(image)

    print()
    print("=" * 55)
    print("           DOCUMENT DETECTION")
    print("=" * 55)
    print("Document Type : UNKNOWN")
    print("Unable to identify the document.")
    print("=" * 55)

    return unknown_result()
