
import os
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from deepface import DeepFace


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Face Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }

    .hero {
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
        background: linear-gradient(
            135deg,
            rgba(70,70,90,0.35),
            rgba(20,20,30,0.55)
        );
        border: 1px solid rgba(255,255,255,0.08);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 0;
    }

    .result-card {
        padding: 22px;
        border-radius: 18px;
        text-align: center;
        min-height: 135px;
        overflow-wrap: anywhere;
        background: rgba(80,80,100,0.18);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 15px;
    }

    .result-card .icon {
        font-size: 30px;
    }

    .result-card .title {
        font-size: 14px;
        opacity: 0.7;
        margin-top: 8px;
    }

    .result-card .value {
        font-size: 24px;
        font-weight: 700;
        margin-top: 5px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 28px;
        margin-bottom: 15px;
    }

    .confidence-card {
        padding: 18px;
        border-radius: 15px;
        background: rgba(80,80,100,0.14);
        border: 1px solid rgba(255,255,255,0.07);
    }

    .emotion-row {
        padding: 8px 12px;
        border-radius: 10px;
        margin-bottom: 5px;
    }

    .footer {
        text-align: center;
        opacity: 0.55;
        margin-top: 45px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URLS = {
    "opencv_face_detector.pbtxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",

    "opencv_face_detector_uint8.pb":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb",

    "age_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",

    "age_net.caffemodel":
        "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/5b60d9f8a8608cdbbcdaaa39bf28f351e8d8553b/model/age_net.caffemodel",

    "gender_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",

    "gender_net.caffemodel":
        "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/gender_net.caffemodel",
}

AGE_LIST = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]

GENDER_LIST = ["Male", "Female"]

MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746,
)


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


# ============================================================
# DOWNLOAD MODEL FILES SAFELY
# ============================================================

def download_model(filename, url):
    path = os.path.join(MODEL_DIR, filename)

    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return path

    temp_path = path + ".download"

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with st.spinner(f"Preparing model: {filename}..."):
            with urllib.request.urlopen(
                request, timeout=180
            ) as response, open(temp_path, "wb") as output:

                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)

        if os.path.getsize(temp_path) <= 1000:
            raise RuntimeError(
                f"Downloaded model is unexpectedly small: {filename}"
            )

        os.replace(temp_path, path)
        return path

    except Exception as exc:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass

        raise RuntimeError(
            f"Could not download {filename}: {exc}"
        ) from exc


# ============================================================
# LOAD OPENCV MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():

    face_proto = download_model(
        "opencv_face_detector.pbtxt",
        MODEL_URLS["opencv_face_detector.pbtxt"],
    )

    face_model = download_model(
        "opencv_face_detector_uint8.pb",
        MODEL_URLS["opencv_face_detector_uint8.pb"],
    )

    age_proto = download_model(
        "age_deploy.prototxt",
        MODEL_URLS["age_deploy.prototxt"],
    )

    age_model = download_model(
        "age_net.caffemodel",
        MODEL_URLS["age_net.caffemodel"],
    )

    gender_proto = download_model(
        "gender_deploy.prototxt",
        MODEL_URLS["gender_deploy.prototxt"],
    )

    gender_model = download_model(
        "gender_net.caffemodel",
        MODEL_URLS["gender_net.caffemodel"],
    )

    face_net = cv2.dnn.readNet(
        face_model,
        face_proto,
    )

    age_net = cv2.dnn.readNet(
        age_model,
        age_proto,
    )

    gender_net = cv2.dnn.readNet(
        gender_model,
        gender_proto,
    )

    return face_net, age_net, gender_net


# ============================================================
# OPENCV FALLBACK FACE DETECTOR
# ============================================================

def detect_faces_opencv(frame, face_net, threshold=0.50):

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        [104, 117, 123],
        swapRB=False,
        crop=False,
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []

    for i in range(detections.shape[2]):

        confidence = float(detections[0, 0, i, 2])

        if confidence < threshold:
            continue

        box = (
            detections[0, 0, i, 3:7]
            * np.array([width, height, width, height])
        )

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        faces.append({
            "x": x1,
            "y": y1,
            "x2": x2,
            "y2": y2,
            "confidence": confidence,
            "area": (x2 - x1) * (y2 - y1),
        })

    faces.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    return faces


# ============================================================
# RETINAFACE FIRST, OPENCV FALLBACK
# ============================================================

def detect_faces(frame, face_net):

    height, width = frame.shape[:2]
    retina_error = None

    try:
        found = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=True,
            align=False,
            grayscale=False,
            normalize_face=False,
        )

        faces = []

        for item in found:

            area = (
                item.get("facial_area")
                or item.get("region")
                or {}
            )

            x = int(safe_float(area.get("x", 0)))
            y = int(safe_float(area.get("y", 0)))
            w = int(safe_float(area.get("w", area.get("width", 0))))
            h = int(safe_float(area.get("h", area.get("height", 0))))

            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            x2 = max(0, min(width, x + w))
            y2 = max(0, min(height, y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            faces.append({
                "x": x1,
                "y": y1,
                "x2": x2,
                "y2": y2,
                "confidence": safe_float(
                    item.get("confidence", 0.0)
                ),
                "area": (x2 - x1) * (y2 - y1),
            })

        if faces:
            faces.sort(
                key=lambda item: item["area"],
                reverse=True,
            )
            return faces, "RetinaFace"

    except Exception as exc:
        retina_error = str(exc)

    # If RetinaFace fails, try OpenCV's detector.
    try:
        faces = detect_faces_opencv(frame, face_net)

        if faces:
            return faces, "OpenCV fallback"

    except Exception as exc:
        raise RuntimeError(
            f"RetinaFace failed: {retina_error}; "
            f"OpenCV fallback failed: {exc}"
        ) from exc

    return [], "RetinaFace/OpenCV"


# ============================================================
# FACE CROPPING
# ============================================================

def padded_crop(
    frame,
    x1,
    y1,
    x2,
    y2,
    padding_ratio=0.10,
):

    height, width = frame.shape[:2]

    padding_x = int((x2 - x1) * padding_ratio)
    padding_y = int((y2 - y1) * padding_ratio)

    left = max(0, x1 - padding_x)
    top = max(0, y1 - padding_y)
    right = min(width, x2 + padding_x)
    bottom = min(height, y2 + padding_y)

    crop = frame[top:bottom, left:right].copy()

    if crop.size == 0:
        raise ValueError("The face crop is empty.")

    return crop


# ============================================================
# AGE AND GENDER ESTIMATION
# ============================================================

def predict_age_gender(face, age_net, gender_net):

    if face is None or face.size == 0:
        return "Unknown", 0.0, "Unavailable", 0.0

    blob = cv2.dnn.blobFromImage(
        face,
        1.0,
        (227, 227),
        MODEL_MEAN_VALUES,
        swapRB=False,
        crop=False,
    )

    gender_net.setInput(blob)
    gender_scores = gender_net.forward()[0]

    gender_index = int(np.argmax(gender_scores))
    gender = GENDER_LIST[gender_index]

    gender_score = float(
        gender_scores[gender_index] * 100
    )

    age_net.setInput(blob)
    age_scores = age_net.forward()[0]

    age_index = int(np.argmax(age_scores))
    age_range = AGE_LIST[age_index]

    age_score = float(
        age_scores[age_index] * 100
    )

    return gender, gender_score, age_range, age_score


# ============================================================
# EXPRESSION ANALYSIS
# ============================================================

def predict_emotion(face):

    try:
        result = DeepFace.analyze(
            img_path=face,
            actions=["emotion"],
            detector_backend="skip",
            enforce_detection=False,
            align=False,
            silent=True,
        )

        if isinstance(result, list):
            result = result[0] if result else {}

        emotions = result.get("emotion", {}) or {}

        expression = str(
            result.get("dominant_emotion", "Unknown")
        )

        score = safe_float(
            emotions.get(expression, 0.0)
        ) if expression != "Unknown" else 0.0

        return expression, score, emotions, None

    except Exception as exc:
        return "Unavailable", 0.0, {}, str(exc)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🤖 AI Face Analysis</h1>
        <p>
            AI-powered Face, Age, Gender and Expression Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    input_method = st.radio(
        "Choose image source:",
        ["📷 Camera", "🖼️ Upload Image"],
    )

    st.divider()

    st.markdown("### 📌 Features")
    st.write("👤 Face Detection")
    st.write("🎂 Age Estimation")
    st.write("⚧ Gender-category Estimate")
    st.write("😊 Expression Analysis")
    st.write("📊 Model Scores")

    st.divider()

    st.info(
        "AI predictions are estimates. Model scores are not "
        "real-world accuracy rates."
    )


# ============================================================
# IMAGE INPUT
# ============================================================

if input_method == "📷 Camera":
    source = st.camera_input("📷 Take a picture")
else:
    source = st.file_uploader(
        "🖼️ Upload an image",
        type=["jpg", "jpeg", "png"],
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if source is not None:

    try:
        pil_image = ImageOps.exif_transpose(
            Image.open(source)
        ).convert("RGB")

        rgb_image = np.asarray(pil_image)
        frame = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

    except Exception as exc:
        st.error("Could not read this image. Please try a JPG or PNG.")
        st.exception(exc)
        st.stop()

    try:
        with st.spinner("🤖 Loading OpenCV models..."):
            face_net, age_net, gender_net = load_models()

    except Exception as exc:
        st.error("AI models could not be loaded.")
        st.exception(exc)
        st.stop()

    try:
        with st.spinner("🔍 Detecting faces..."):
            faces, detector_name = detect_faces(
                frame,
                face_net,
            )

    except Exception as exc:
        st.error("Face detection failed.")
        st.exception(exc)
        st.stop()

    if not faces:
        st.warning(
            "No clear face detected. Please try a well-lit, "
            "front-facing photo."
        )
        st.stop()

    # Select and analyze only the largest detected face.
    selected_face = faces[0]

    x1 = selected_face["x"]
    y1 = selected_face["y"]
    x2 = selected_face["x2"]
    y2 = selected_face["y2"]

    try:
        # Tight crop is used for age and gender estimates.
        age_gender_crop = frame[y1:y2, x1:x2].copy()

        # Slightly padded crop is used for expression analysis.
        expression_crop = padded_crop(
            frame,
            x1,
            y1,
            x2,
            y2,
            padding_ratio=0.10,
        )

        if age_gender_crop.size == 0:
            raise ValueError("Face crop is empty.")

        with st.spinner("👤 Estimating age and gender..."):
            (
                gender,
                gender_score,
                age_range,
                age_score,
            ) = predict_age_gender(
                age_gender_crop,
                age_net,
                gender_net,
            )

        with st.spinner("😊 Estimating expression..."):
            (
                expression,
                expression_score,
                emotions,
                expression_error,
            ) = predict_emotion(expression_crop)

    except Exception as exc:
        st.error("Face analysis failed.")
        st.exception(exc)
        st.stop()

    # ========================================================
    # DRAW ONE FACE BOX
    # ========================================================

    result_frame = frame.copy()

    cv2.rectangle(
        result_frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    gender_short = (
        "Male?" if gender == "Male"
        else "Female?" if gender == "Female"
        else "Unknown"
    )

    label = (
        f"{gender_short} | Age {age_range} | "
        f"{expression.capitalize()}"
    )

    font_scale = max(
        0.45,
        min(0.72, frame.shape[1] / 1400.0),
    )

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        2,
    )

    text_y = max(text_height + 12, y1 - 8)

    text_x2 = min(
        frame.shape[1] - 1,
        x1 + text_width + 10,
    )

    cv2.rectangle(
        result_frame,
        (x1, text_y - text_height - 8),
        (text_x2, text_y + baseline),
        (10, 18, 32),
        -1,
    )

    cv2.putText(
        result_frame,
        label,
        (x1 + 5, text_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # ========================================================
    # RESULT IMAGE
    # ========================================================

    st.markdown(
        '<div class="section-title">🔍 Face Detection Result</div>',
        unsafe_allow_html=True,
    )

    st.image(
        cv2.cvtColor(
            result_frame,
            cv2.COLOR_BGR2RGB,
        ),
        width="stretch",
    )

    st.caption(
        f"Detector: {detector_name} • "
        f"Faces detected: {len(faces)} • "
        "Largest face only is analyzed."
    )

    # ========================================================
    # RESULT CARDS
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Analysis Result</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="icon">👤</div>
                <div class="title">Gender Category (Estimate)</div>
                <div class="value">{gender}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="icon">🎂</div>
                <div class="title">Estimated Age Range</div>
                <div class="value">{age_range}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="icon">😊</div>
                <div class="title">Expression Estimate</div>
                <div class="value">{expression.capitalize()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # MODEL SCORES
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Model Scores (Not Accuracy)</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="confidence-card">',
            unsafe_allow_html=True,
        )
        st.write(f"**👤 Gender:** {gender_score:.1f}% model score")
        st.progress(
            min(max(gender_score / 100.0, 0.0), 1.0)
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="confidence-card">',
            unsafe_allow_html=True,
        )
        st.write(f"**🎂 Age:** {age_score:.1f}% model score")
        st.progress(
            min(max(age_score / 100.0, 0.0), 1.0)
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(
            '<div class="confidence-card">',
            unsafe_allow_html=True,
        )
        st.write(f"**😊 Expression:** {expression_score:.1f}% model score")
        st.progress(
            min(max(expression_score / 100.0, 0.0), 1.0)
        )
        st.markdown("</div>", unsafe_allow_html=True)

    detector_score = safe_float(
        selected_face.get("confidence", 0.0)
    )

    if detector_score <= 1.0:
        detector_score *= 100.0

    st.write(
        f"**🔎 Face Detection Score:** {detector_score:.1f}% "
        "(not prediction accuracy)"
    )

    # ========================================================
    # SHOW MODEL INPUT CROPS
    # ========================================================

    with st.expander("View face crops sent to models"):

        crop_col1, crop_col2 = st.columns(2)

        with crop_col1:
            st.caption("Age and gender input")
            st.image(
                cv2.cvtColor(
                    age_gender_crop,
                    cv2.COLOR_BGR2RGB,
                ),
                width=220,
            )

        with crop_col2:
            st.caption("Expression input")
            st.image(
                cv2.cvtColor(
                    expression_crop,
                    cv2.COLOR_BGR2RGB,
                ),
                width=220,
            )

    # ========================================================
    # EXPRESSION DETAILS
    # ========================================================

    if emotions:

        st.markdown(
            '<div class="section-title">😊 Expression Details</div>',
            unsafe_allow_html=True,
        )

        emotion_names = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]

        emotion_col1, emotion_col2 = st.columns(2)

        for index, emotion in enumerate(emotion_names):

            if emotion not in emotions:
                continue

            value = max(
                0.0,
                min(100.0, safe_float(emotions[emotion])),
            )

            current_col = (
                emotion_col1 if index % 2 == 0
                else emotion_col2
            )

            with current_col:
                st.markdown(
                    f"""
                    <div class="emotion-row">
                        <b>{emotion.capitalize()}</b> — {value:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(value / 100.0)

    elif expression_error:
        st.warning(
            f"Expression analysis unavailable: {expression_error}"
        )

    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">📥 Analysis Report</div>',
        unsafe_allow_html=True,
    )

    report = f"""
AI FACE ANALYSIS REPORT
=======================

Gender Category (Estimate): {gender}
Gender Model Score: {gender_score:.1f}%

Estimated Age Range: {age_range}
Age Model Score: {age_score:.1f}%

Expression Estimate: {expression.capitalize()}
Expression Model Score: {expression_score:.1f}%

Detector: {detector_name}
Face Detection Score: {detector_score:.1f}%

NOTE:
AI predictions are estimates, not verified facts.
Model scores are not real-world accuracy rates.
The age model provides broad age ranges, not exact age.
"""

    st.download_button(
        label="⬇️ Download Analysis Report",
        data=report,
        file_name="AI_Face_Analysis_Report.txt",
        mime="text/plain",
        width="stretch",
    )

    st.warning(
        "AI predictions are estimates. Age is shown as a broad range, "
        "not exact age. Gender-category and expression predictions can "
        "also be wrong, even when model scores are high."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🤖 AI Face Analysis System<br>
        <small>Powered by Computer Vision &amp; AI | Team Night Furry</small>
    </div>
    """,
    unsafe_allow_html=True,
)