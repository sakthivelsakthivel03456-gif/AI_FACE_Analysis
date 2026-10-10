
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from deepface import DeepFace

st.set_page_config(page_title="AI Face Analysis", page_icon="🤖", layout="wide")

st.markdown(
    """
    <h1 style="text-align:center;">🤖 AI Face Analysis System</h1>
    <p style="text-align:center;color:#999;">
        Face Detection • Approximate Age • Gender-category Estimate • Expression
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    input_method = st.radio("Choose image source:", ["📷 Camera", "🖼️ Upload Image"])
    st.divider()
    st.subheader("📌 Features")
    st.write("👤 Face Detection")
    st.write("🎂 Approximate Age Estimation")
    st.write("⚧️ Gender-category Estimate")
    st.write("😊 Facial Expression Estimate")
    st.info(
        "Predictions are estimates, not verified facts. Model scores are not "
        "real-world accuracy rates, and these predictions can be wrong."
    )

if input_method == "📷 Camera":
    source = st.camera_input("Take a picture")
else:
    source = st.file_uploader("Upload a clear face photo", type=["jpg", "jpeg", "png"])


def as_uint8_face(face):
    """Convert a DeepFace crop to a valid uint8 BGR array."""
    arr = np.asarray(face)
    if arr.size == 0:
        raise ValueError("Empty face crop")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if np.nanmax(arr) <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Face crop is not a 3-channel image")
    return arr


def safe_float(value, default=0.0):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def extract_region(face_item, image_width, image_height):
    area = face_item.get("facial_area") or face_item.get("region") or {}
    x = int(safe_float(area.get("x", 0)))
    y = int(safe_float(area.get("y", 0)))
    w = int(safe_float(area.get("w", area.get("width", 0))))
    h = int(safe_float(area.get("h", area.get("height", 0))))
    x1 = max(0, min(image_width - 1, x))
    y1 = max(0, min(image_height - 1, y))
    x2 = max(0, min(image_width, x + w))
    y2 = max(0, min(image_height, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def as_results(value):
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def gender_label(result):
    raw = str(result.get("dominant_gender", "Unknown")).strip().lower()
    if raw in ("man", "male"):
        return "Male-category estimate"
    if raw in ("woman", "female"):
        return "Female-category estimate"
    return "Unknown"


def age_label(result):
    age = result.get("age")
    try:
        age = int(round(float(age)))
        if age < 0:
            return "Unavailable"
        return f"About {age} years (estimate)"
    except (TypeError, ValueError):
        return "Unavailable"


def score_percent(value):
    """DeepFace face-confidence is commonly 0..1; score dictionaries use 0..100."""
    value = safe_float(value, -1.0)
    if value < 0:
        return None
    if value <= 1.0:
        value *= 100.0
    return max(0.0, min(100.0, value))


if source is not None:
    try:
        pil_img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
        rgb_image = np.asarray(pil_img)
        # DeepFace accepts OpenCV-style BGR NumPy images.
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    except Exception as err:
        st.error("Image open panna mudiyala da. JPG/PNG photo try pannu.")
        st.exception(err)
        st.stop()

    # Detect faces, then select ONLY the largest face for analysis/display.
    # This prevents multiple boxes and overlapping labels in group photos.
    extracted_faces = None
    chosen_backend = None
    detector_errors = []

    with st.spinner("🔎 Main face detect pannudhu..."):
        for backend in ("retinaface", "opencv", "ssd"):
            try:
                found = DeepFace.extract_faces(
                    img_path=bgr_image,
                    detector_backend=backend,
                    enforce_detection=True,
                    align=True,
                    expand_percentage=0,
                    grayscale=False,
                    color_face="bgr",
                    normalize_face=False,
                )
                if found:
                    extracted_faces = found
                    chosen_backend = backend
                    break
            except Exception as err:
                detector_errors.append(
                    f"{backend}: {type(err).__name__}: {err}"
                )

    if not extracted_faces:
        st.error(
            "Face detect panna mudiyala da. Clear-ah, front-facing photo use pannu."
        )
        with st.expander("Technical details"):
            for detail in detector_errors:
                st.write(detail)
        st.stop()

    height, width = rgb_image.shape[:2]

    # Select the largest valid face region. Other faces won't be boxed or analyzed.
    valid_faces = []
    for extracted in extracted_faces:
        region = extract_region(extracted, width, height)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        area = (x2 - x1) * (y2 - y1)
        valid_faces.append((area, extracted, region))

    if not valid_faces:
        st.error("Detected face-ku valid area kidaikkala da. Vera photo try pannu.")
        st.stop()

    valid_faces.sort(key=lambda item: item[0], reverse=True)
    _, selected_face, (x1, y1, x2, y2) = valid_faces[0]

    try:
        with st.spinner("🧠 Main face-ah analyze pannudhu..."):
            face_crop = as_uint8_face(selected_face.get("face"))
            # The crop is already detected/aligned, so skip a second detection.
            raw = DeepFace.analyze(
                img_path=face_crop,
                actions=["age", "gender", "emotion"],
                detector_backend="skip",
                enforce_detection=False,
                align=False,
                silent=True,
            )
            analyses = as_results(raw)
            if not analyses:
                raise RuntimeError("Model returned no analysis result")
            item = analyses[0]
    except Exception as err:
        st.error("Main face analyze panna mudiyala da.")
        st.exception(err)
        st.stop()

    gender = gender_label(item)
    age = age_label(item)
    emotion = str(item.get("dominant_emotion", "Unknown")).capitalize()

    # Draw ONE rectangle and ONE short label only.
    marked = bgr_image.copy()
    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 220, 90), 3)

    if gender == "Male-category estimate":
        short_gender = "Male?"
    elif gender == "Female-category estimate":
        short_gender = "Female?"
    else:
        short_gender = "Unknown"

    short_age = age.replace(" (estimate)", "")
    label = f"{short_gender} | {short_age} | {emotion}"
    scale = max(0.45, min(0.72, width / 1400.0))
    (tw, th), base = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
    )
    text_y = max(th + 12, y1 - 6)
    tx2 = min(width - 1, x1 + tw + 10)
    cv2.rectangle(
        marked,
        (x1, text_y - th - 8),
        (tx2, text_y + base),
        (10, 18, 32),
        -1,
    )
    cv2.putText(
        marked,
        label,
        (x1 + 5, text_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    raw_confidence = selected_face.get(
        "confidence", selected_face.get("face_confidence")
    )
    detector_score = (
        score_percent(raw_confidence)
        if raw_confidence is not None
        else None
    )

    st.caption(
        f"Face detector used: {chosen_backend}. "
        f"Faces found: {len(extracted_faces)}; analyzing the largest face only."
    )

    st.subheader("🔍 Face Detection Result")
    st.image(
        cv2.cvtColor(marked, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )

    st.subheader("📊 Analysis Result")
    with st.container(border=True):
        st.markdown("### 👤 Main Face")
        c1, c2, c3 = st.columns(3)
        c1.metric("Gender category (estimate)", gender)
        c2.metric("Approximate age", age)
        c3.metric("Expression estimate", emotion)

        if detector_score is not None:
            st.caption(
                f"Face detection score: {detector_score:.1f}% "
                "(not prediction accuracy)"
            )

        with st.expander("View face crop sent to the analysis model"):
            st.image(
                cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
                caption="Selected largest face crop",
                width=220,
            )

        gender_scores = item.get("gender", {}) or {}
        if gender_scores:
            st.markdown("**Gender-category model scores (not accuracy)**")
            for name, val in gender_scores.items():
                st.write(f"{name}: {safe_float(val):.1f}% model score")

        emotion_scores = item.get("emotion", {}) or {}
        if emotion_scores:
            st.markdown("**Expression model scores (not verified emotions)**")
            for name in (
                "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
            ):
                if name not in emotion_scores:
                    continue
                value = max(0.0, min(100.0, safe_float(emotion_scores[name])))
                st.write(f"{name.capitalize()}: {value:.1f}% model score")
                st.progress(int(round(value)))

    st.info(
        "Only the largest detected face is analyzed. Age, gender-category, and "
        "expression values are model estimates and may be incorrect, even when "
        "model scores are high. A detector score measures face detection, not "
        "prediction accuracy."
    )
else:
    st.markdown("### 👋 Welcome!\nUse the camera or upload a clear face photo to start.")

st.markdown(
    "<p style='text-align:center;color:#888;margin-top:35px;'>"
    "🤖 AI Face Analysis System | Team Night Furry</p>",
    unsafe_allow_html=True,
)