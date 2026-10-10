
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from deepface import DeepFace

st.set_page_config(
    page_title="AI Face Analysis",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <h1 style="text-align:center;">🤖 AI Face Analysis System</h1>
    <p style="text-align:center;color:#999;">
        Face Detection • Estimated Age • Gender Category • Expression
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    source_type = st.radio(
        "Choose image source:",
        ["📷 Camera", "🖼️ Upload Image"],
    )
    st.divider()
    st.subheader("📌 Features")
    st.write("👤 Face Detection")
    st.write("🎂 Approximate Age Estimation")
    st.write("⚧️ Gender-category Estimate")
    st.write("😊 Facial Expression Estimate")
    st.info(
        "AI results are estimates. Model scores are not "
        "real-world accuracy rates."
    )


def normalize_results(value):
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def face_area(item):
    region = item.get("region") or item.get("facial_area") or {}
    width = safe_float(region.get("w", region.get("width", 0)))
    height = safe_float(region.get("h", region.get("height", 0)))
    return max(0, width) * max(0, height)


def get_region(item, width, height):
    region = item.get("region") or item.get("facial_area") or {}
    x = int(safe_float(region.get("x", 0)))
    y = int(safe_float(region.get("y", 0)))
    w = int(safe_float(region.get("w", region.get("width", 0))))
    h = int(safe_float(region.get("h", region.get("height", 0))))

    x1 = max(0, min(width - 1, x))
    y1 = max(0, min(height - 1, y))
    x2 = max(0, min(width, x + w))
    y2 = max(0, min(height, y + h))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def gender_text(result):
    raw = str(result.get("dominant_gender", "Unknown")).strip().lower()

    if raw in ("man", "male"):
        return "Male-category estimate"
    if raw in ("woman", "female"):
        return "Female-category estimate"
    return "Unknown"


def estimated_age(result):
    try:
        age = int(round(float(result.get("age"))))
        return f"About {age} years (estimate)"
    except (TypeError, ValueError):
        return "Unavailable"


def score_to_percent(value):
    if value is None:
        return None

    score = safe_float(value, -1)
    if score < 0:
        return None

    if score <= 1.0:
        score *= 100

    return max(0.0, min(100.0, score))


if source_type == "📷 Camera":
    uploaded = st.camera_input("Take a picture")
else:
    uploaded = st.file_uploader(
        "Upload a clear face photo",
        type=["jpg", "jpeg", "png"],
    )


if uploaded is not None:
    try:
        pil_image = ImageOps.exif_transpose(
            Image.open(uploaded)
        ).convert("RGB")

        rgb_image = np.asarray(pil_image)
        bgr_image = cv2.cvtColor(
            rgb_image, cv2.COLOR_RGB2BGR
        )

    except Exception as exc:
        st.error("Image open panna mudiyala da. JPG/PNG try pannu.")
        st.exception(exc)
        st.stop()

    results = None
    selected_backend = None
    detector_errors = []

    # Try stronger face detection first, with fallbacks.
    with st.spinner("🔎 Face detect panni analyze pannudhu..."):
        for backend in ("retinaface", "opencv", "ssd"):
            try:
                raw = DeepFace.analyze(
                    img_path=bgr_image,
                    actions=["age", "gender", "emotion"],
                    detector_backend=backend,
                    enforce_detection=True,
                    align=True,
                    silent=True,
                )

                candidate = normalize_results(raw)

                if candidate:
                    results = candidate
                    selected_backend = backend
                    break

            except Exception as exc:
                detector_errors.append(
                    f"{backend}: {type(exc).__name__}: {exc}"
                )

    if not results:
        st.error(
            "Face detect panna mudiyala da. "
            "Clear-ah, front-facing photo try pannu."
        )

        with st.expander("Technical details"):
            for error in detector_errors:
                st.write(error)

        st.stop()

    # Keep only the largest face for display and results.
    results.sort(key=face_area, reverse=True)
    main_result = results[0]

    img_h, img_w = bgr_image.shape[:2]
    region = get_region(main_result, img_w, img_h)

    if region is None:
        st.error("Valid face box kidaikkala da. Vera photo try pannu.")
        st.stop()

    x1, y1, x2, y2 = region

    gender = gender_text(main_result)
    age = estimated_age(main_result)
    expression = str(
        main_result.get("dominant_emotion", "Unknown")
    ).capitalize()

    # Draw only one green box.
    marked = bgr_image.copy()
    cv2.rectangle(
        marked, (x1, y1), (x2, y2), (0, 255, 0), 3
    )

    short_gender = (
        "Male?" if gender.startswith("Male")
        else "Female?" if gender.startswith("Female")
        else "Unknown"
    )

    label = (
        f"{short_gender} | "
        f"{age.replace(' (estimate)', '')} | "
        f"{expression}"
    )

    font_scale = max(0.45, min(0.72, img_w / 1400.0))
    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        2,
    )

    text_y = max(text_h + 12, y1 - 8)
    text_x2 = min(img_w - 1, x1 + text_w + 10)

    cv2.rectangle(
        marked,
        (x1, text_y - text_h - 8),
        (text_x2, text_y + baseline),
        (10, 18, 32),
        -1,
    )

    cv2.putText(
        marked,
        label,
        (x1 + 5, text_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    st.caption(
        f"Detector: {selected_backend} • "
        f"Faces detected: {len(results)} • "
        "Largest face only is displayed."
    )

    st.subheader("🔍 Face Detection Result")
    st.image(
        cv2.cvtColor(marked, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )

    st.subheader("📊 Analysis Result — Main Face Only")

    col1, col2, col3 = st.columns(3)
    col1.metric("Gender category (estimate)", gender)
    col2.metric("Approximate age", age)
    col3.metric("Expression estimate", expression)

    face_score = score_to_percent(
        main_result.get("face_confidence")
    )

    if face_score is not None:
        st.caption(
            f"Face detector score: {face_score:.1f}% "
            "(not prediction accuracy)."
        )

    # Show scores from DeepFace without calling them accuracy.
    gender_scores = main_result.get("gender", {}) or {}
    emotion_scores = main_result.get("emotion", {}) or {}

    if gender_scores:
        with st.expander("Gender-category model scores"):
            for name, value in gender_scores.items():
                st.write(
                    f"{name}: {safe_float(value):.1f}% model score"
                )

    if emotion_scores:
        with st.expander("Expression model scores"):
            for name in (
                "angry", "disgust", "fear", "happy",
                "sad", "surprise", "neutral"
            ):
                if name in emotion_scores:
                    value = max(
                        0.0,
                        min(100.0, safe_float(emotion_scores[name]))
                    )
                    st.write(
                        f"{name.capitalize()}: "
                        f"{value:.1f}% model score"
                    )
                    st.progress(int(round(value)))

    with st.expander("View selected face crop"):
        st.image(
            cv2.cvtColor(
                bgr_image[y1:y2, x1:x2],
                cv2.COLOR_BGR2RGB,
            ),
            width=240,
        )

    st.info(
        "Age is an estimate, not an exact age. Gender-category and "
        "expression predictions can also be wrong, even when model "
        "scores are high."
    )

else:
    st.markdown(
        "### 👋 Welcome!\n"
        "Use the camera or upload a clear face photo to start."
    )


st.markdown(
    "<p style='text-align:center;color:#888;margin-top:35px;'>"
    "🤖 AI Face Analysis System | Team Night Furry</p>",
    unsafe_allow_html=True,
)