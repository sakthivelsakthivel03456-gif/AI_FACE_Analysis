
import hashlib

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
        "AI predictions are estimates. Model scores are not "
        "real-world accuracy rates."
    )

if source_type == "📷 Camera":
    uploaded = st.camera_input("Take a picture")
else:
    uploaded = st.file_uploader(
        "Upload a clear face photo",
        type=["jpg", "jpeg", "png"],
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
    raw = str(
        result.get("dominant_gender", "Unknown")
    ).strip().lower()

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


if uploaded is not None:
    image_hash = hashlib.sha256(
        uploaded.getvalue()
    ).hexdigest()

    # Clear old confirmations when a different image is uploaded.
    if st.session_state.get("_active_image_hash") != image_hash:
        st.session_state["_active_image_hash"] = image_hash
        st.session_state["_confirmed_details"] = {}
        st.session_state.pop("_analysis_hash", None)
        st.session_state.pop("_analysis_results", None)
        st.session_state.pop("_analysis_backend", None)

    try:
        pil_image = ImageOps.exif_transpose(
            Image.open(uploaded)
        ).convert("RGB")

        rgb_image = np.asarray(pil_image)
        bgr_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

    except Exception as exc:
        st.error("Image open panna mudiyala da. JPG/PNG try pannu.")
        st.exception(exc)
        st.stop()

    # Reuse AI analysis when only the confirmation form changes.
    raw_results = st.session_state.get("_analysis_results")
    selected_backend = st.session_state.get("_analysis_backend")
    detector_errors = []

    if (
        st.session_state.get("_analysis_hash") != image_hash
        or not raw_results
    ):
        raw_results = None
        selected_backend = None

        with st.spinner("🔎 Face detect panni analyze pannudhu..."):
            for backend in ("retinaface", "opencv", "ssd"):
                try:
                    candidate = DeepFace.analyze(
                        img_path=bgr_image,
                        actions=["age", "gender", "emotion"],
                        detector_backend=backend,
                        enforce_detection=True,
                        align=True,
                        silent=True,
                    )

                    candidate_results = normalize_results(candidate)

                    if candidate_results:
                        raw_results = candidate_results
                        selected_backend = backend
                        break

                except Exception as exc:
                    detector_errors.append(
                        f"{backend}: {type(exc).__name__}: {exc}"
                    )

        if raw_results:
            st.session_state["_analysis_hash"] = image_hash
            st.session_state["_analysis_results"] = raw_results
            st.session_state["_analysis_backend"] = selected_backend

    if not raw_results:
        st.error(
            "Face detect/analyze panna mudiyala da. "
            "Clear front-facing photo try pannu."
        )

        with st.expander("Technical details"):
            for error in detector_errors:
                st.write(error)

        st.stop()

    # Select the largest face only.
    raw_results.sort(key=face_area, reverse=True)
    main_result = raw_results[0]

    img_h, img_w = bgr_image.shape[:2]
    region = get_region(main_result, img_w, img_h)

    if region is None:
        st.error("Valid face box kidaikkala da. Vera photo try pannu.")
        st.stop()

    x1, y1, x2, y2 = region

    ai_gender = gender_text(main_result)
    ai_age = estimated_age(main_result)
    ai_expression = str(
        main_result.get("dominant_emotion", "Unknown")
    ).capitalize()

    # Draw one box on the largest detected face.
    marked = bgr_image.copy()
    cv2.rectangle(
        marked,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    short_gender = (
        "Male?" if ai_gender.startswith("Male")
        else "Female?" if ai_gender.startswith("Female")
        else "Unknown"
    )

    label = (
        f"{short_gender} | "
        f"{ai_age.replace(' (estimate)', '')} | "
        f"{ai_expression}"
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
        f"Faces detected: {len(raw_results)} • "
        "Largest face only is displayed."
    )

    st.subheader("🔍 Face Detection Result")
    st.image(
        cv2.cvtColor(marked, cv2.COLOR_BGR2RGB),
        width="stretch",
    )

    # -----------------------------------------------
    # AI PREDICTIONS
    # -----------------------------------------------

    st.subheader("🤖 AI Prediction — Main Face Only")

    col1, col2, col3 = st.columns(3)
    col1.metric("Gender category (estimate)", ai_gender)
    col2.metric("Approximate age", ai_age)
    col3.metric("Expression estimate", ai_expression)

    face_score = score_to_percent(
        main_result.get("face_confidence")
    )

    if face_score is not None:
        st.caption(
            f"Face detector score: {face_score:.1f}% "
            "(not prediction accuracy)."
        )

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
                        min(100.0, safe_float(emotion_scores[name])),
                    )
                    st.write(
                        f"{name.capitalize()}: {value:.1f}% model score"
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
        "Age, gender-category and expression are AI estimates. "
        "Model scores are not real-world accuracy rates."
    )

    # -----------------------------------------------
    # USER-CONFIRMED DETAILS
    # -----------------------------------------------

    st.subheader("✅ Confirmed Details (Optional)")

    st.caption(
        "AI prediction mela irukkura section-la thaniya irukkum. "
        "Unakku therinja, permission irukkura details mattum enter pannu. "
        "Indha code confirmed details-ai database-la save pannaadhu."
    )

    with st.form(key=f"confirmed_details_form_{image_hash[:12]}"):
        age_known = st.checkbox(
            "Actual age enakku theriyum; confirm panna virumburen",
            key=f"age_known_{image_hash[:12]}",
        )

        actual_age = st.number_input(
            "Confirmed age (years)",
            min_value=0,
            max_value=120,
            value=18,
            step=1,
            help="Age confirm panna therinja mattum checkbox select pannu.",
            key=f"confirmed_age_{image_hash[:12]}",
        )

        confirmed_gender = st.selectbox(
            "Confirmed gender (self-reported)",
            [
                "Not provided",
                "Woman",
                "Man",
                "Non-binary",
                "Another identity",
                "Prefer not to say",
            ],
            key=f"confirmed_gender_{image_hash[:12]}",
        )

        confirmed_expression = st.selectbox(
            "Person-confirmed expression description",
            [
                "Not provided",
                "Happy",
                "Sad",
                "Angry",
                "Neutral",
                "Surprised",
                "Other",
                "Prefer not to say",
            ],
            key=f"confirmed_expression_{image_hash[:12]}",
        )

        save_confirmed = st.form_submit_button(
            "Save confirmed details"
        )

    if save_confirmed:
        confirmed = {}

        if age_known:
            confirmed["Age"] = (
                f"{int(actual_age)} years (user-confirmed)"
            )

        if confirmed_gender != "Not provided":
            confirmed["Gender"] = confirmed_gender

        if confirmed_expression != "Not provided":
            confirmed["Expression description"] = confirmed_expression

        st.session_state["_confirmed_details"] = confirmed

    confirmed = st.session_state.get("_confirmed_details", {})

    if confirmed:
        st.markdown(
            "### ✅ Confirmed Details — Separate from AI Predictions"
        )

        for field, value in confirmed.items():
            st.write(f"**{field}:** {value}")
    else:
        st.caption("Innum confirmed details enter pannala.")

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