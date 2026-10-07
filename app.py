import os
import urllib.request

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Face Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

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
        font-size: 25px;
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
    unsafe_allow_html=True
)


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# MODEL FILES
# ============================================================

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
        "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/gender_net.caffemodel"
}


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def download_model(filename, url):

    path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path

    try:

        with st.spinner(f"Downloading {filename}..."):

            urllib.request.urlretrieve(
                url,
                path
            )

        if os.path.getsize(path) <= 1000:
            raise RuntimeError(
                f"Downloaded model is too small: {filename}"
            )

        return path

    except Exception as e:

        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

        raise RuntimeError(
            f"Could not download {filename}: {e}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():

    face_proto = download_model(
        "opencv_face_detector.pbtxt",
        MODEL_URLS["opencv_face_detector.pbtxt"]
    )

    face_model = download_model(
        "opencv_face_detector_uint8.pb",
        MODEL_URLS["opencv_face_detector_uint8.pb"]
    )

    age_proto = download_model(
        "age_deploy.prototxt",
        MODEL_URLS["age_deploy.prototxt"]
    )

    age_model = download_model(
        "age_net.caffemodel",
        MODEL_URLS["age_net.caffemodel"]
    )

    gender_proto = download_model(
        "gender_deploy.prototxt",
        MODEL_URLS["gender_deploy.prototxt"]
    )

    gender_model = download_model(
        "gender_net.caffemodel",
        MODEL_URLS["gender_net.caffemodel"]
    )

    face_net = cv2.dnn.readNet(
        face_model,
        face_proto
    )

    age_net = cv2.dnn.readNet(
        age_model,
        age_proto
    )

    gender_net = cv2.dnn.readNet(
        gender_model,
        gender_proto
    )

    return face_net, age_net, gender_net


# ============================================================
# AGE / GENDER CLASSES
# ============================================================

AGE_LIST = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]

GENDER_LIST = [
    "Male",
    "Female"
]

MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746
)


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(frame, face_net):

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        [104, 117, 123],
        swapRB=False,
        crop=False
    )

    face_net.setInput(blob)

    detections = face_net.forward()

    faces = []

    for i in range(detections.shape[2]):

        confidence = float(
            detections[0, 0, i, 2]
        )

        if confidence < 0.50:
            continue

        box = (
            detections[0, 0, i, 3:7]
            * np.array([w, h, w, h])
        )

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        faces.append(
            {
                "x": x1,
                "y": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence
            }
        )

    faces.sort(
        key=lambda item:
        (item["x2"] - item["x"])
        *
        (item["y2"] - item["y"]),
        reverse=True
    )

    return faces


# ============================================================
# AGE + GENDER
# ============================================================

def predict_age_gender(face, age_net, gender_net):

    blob = cv2.dnn.blobFromImage(
        face,
        1.0,
        (227, 227),
        MODEL_MEAN_VALUES,
        swapRB=False
    )

    gender_net.setInput(blob)

    gender_predictions = gender_net.forward()

    gender_index = int(
        gender_predictions[0].argmax()
    )

    gender = GENDER_LIST[gender_index]

    gender_confidence = float(
        gender_predictions[0][gender_index] * 100
    )

    age_net.setInput(blob)

    age_predictions = age_net.forward()

    age_index = int(
        age_predictions[0].argmax()
    )

    age_range = AGE_LIST[age_index]

    age_confidence = float(
        age_predictions[0][age_index] * 100
    )

    return (
        gender,
        gender_confidence,
        age_range,
        age_confidence
    )


# ============================================================
# EMOTION
# ============================================================

def predict_emotion(face):

    try:

        result = DeepFace.analyze(
            img_path=face,
            actions=["emotion"],
            detector_backend="skip",
            enforce_detection=False,
            align=True,
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result.get(
            "emotion",
            {}
        )

        dominant_emotion = result.get(
            "dominant_emotion",
            "Unknown"
        )

        confidence = float(
            emotions.get(
                dominant_emotion,
                0.0
            )
        )

        return (
            dominant_emotion,
            confidence,
            emotions
        )

    except Exception:
        return (
            "Unknown",
            0.0,
            {}
        )


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
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    input_method = st.radio(
        "Choose image source:",
        [
            "📷 Camera",
            "🖼️ Upload Image"
        ]
    )

    st.divider()

    st.markdown("### 📌 Features")

    st.write("👤 Face Detection")
    st.write("🎂 Age Estimation")
    st.write("⚧ Gender Estimation")
    st.write("😊 Expression Analysis")
    st.write("📊 Confidence Analysis")

    st.divider()

    st.info(
        "AI predictions are estimates. "
        "Results may vary depending on lighting, "
        "image quality and face angle."
    )


# ============================================================
# IMAGE INPUT
# ============================================================

source = None

if input_method == "📷 Camera":

    source = st.camera_input(
        "📷 Take a picture"
    )

else:

    source = st.file_uploader(
        "🖼️ Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


# ============================================================
# ANALYSIS
# ============================================================

if source is not None:

    image_bytes = source.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        st.error(
            "❌ Could not read the image."
        )

        st.stop()

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    try:

        with st.spinner(
            "🤖 Loading AI models..."
        ):

            (
                face_net,
                age_net,
                gender_net
            ) = load_models()

    except Exception as e:

        st.error(
            "❌ AI models could not be loaded."
        )

        st.code(str(e))

        st.stop()

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    with st.spinner(
        "🔍 Detecting face..."
    ):

        faces = detect_faces(
            frame,
            face_net
        )

    if not faces:

        st.warning(
            "⚠️ No clear face detected."
        )

        st.info(
            "Please use a clear photo with "
            "the complete face visible."
        )

        st.stop()

    selected_face = faces[0]

    x1 = selected_face["x"]
    y1 = selected_face["y"]
    x2 = selected_face["x2"]
    y2 = selected_face["y2"]

    detection_confidence = (
        selected_face["confidence"] * 100
    )

    # --------------------------------------------------------
    # CROP FACE
    # --------------------------------------------------------

    face = frame[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:

        st.error(
            "❌ Unable to crop detected face."
        )

        st.stop()

    # --------------------------------------------------------
    # AGE + GENDER
    # --------------------------------------------------------

    with st.spinner(
        "👤 Predicting age and gender..."
    ):

        (
            gender,
            gender_confidence,
            age_range,
            age_confidence
        ) = predict_age_gender(
            face,
            age_net,
            gender_net
        )

    # --------------------------------------------------------
    # EXPRESSION
    # --------------------------------------------------------

    with st.spinner(
        "😊 Detecting expression..."
    ):

        (
            expression,
            expression_confidence,
            emotions
        ) = predict_emotion(face)

    # ========================================================
    # DRAW RESULT
    # ========================================================

    result_frame = frame.copy()

    cv2.rectangle(
        result_frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    label = (
        f"{gender} | Age {age_range} | "
        f"{expression.capitalize()}"
    )

    label_y = max(
        30,
        y1 - 12
    )

    cv2.putText(
        result_frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    result_rgb = cv2.cvtColor(
        result_frame,
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # RESULT IMAGE
    # ========================================================

    st.markdown(
        '<div class="section-title">🔍 Face Detection Result</div>',
        unsafe_allow_html=True
    )

    st.image(
        result_rgb,
        width="stretch"
    )

    # ========================================================
    # MAIN RESULT CARDS
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Analysis Result</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="icon">👤</div>
                <div class="title">Gender</div>
                <div class="value">{gender}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="icon">🎂</div>
                <div class="title">Estimated Age</div>
                <div class="value">{age_range}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="icon">😊</div>
                <div class="title">Expression</div>
                <div class="value">
                    {expression.capitalize()}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # MODEL CONFIDENCE
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Model Confidence</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            '<div class="confidence-card">',
            unsafe_allow_html=True
        )

        st.write(
            f"**👤 Gender:** "
            f"{gender_confidence:.1f}%"
        )

        st.progress(
            min(
                max(
                    gender_confidence / 100,
                    0
                ),
                1
            )
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="confidence-card">',
            unsafe_allow_html=True
        )

        st.write(
            f"**🎂 Age:** "
            f"{age_confidence:.1f}%"
        )

        st.progress(
            min(
                max(
                    age_confidence / 100,
                    0
                ),
                1
            )
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            '<div class="confidence-card">',
            unsafe_allow_html=True
        )

        st.write(
            f"**😊 Expression:** "
            f"{expression_confidence:.1f}%"
        )

        st.progress(
            min(
                max(
                    expression_confidence / 100,
                    0
                ),
                1
            )
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ========================================================
    # DETECTION CONFIDENCE
    # ========================================================

    st.write("")

    st.markdown(
        f"**🔎 Face Detection Confidence:** "
        f"{detection_confidence:.1f}%"
    )

    st.progress(
        min(
            max(
                detection_confidence / 100,
                0
            ),
            1
        )
    )

    # ========================================================
    # EXPRESSION DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">😊 Expression Details</div>',
        unsafe_allow_html=True
    )

    emotion_names = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral"
    ]

    emotion_col1, emotion_col2 = st.columns(2)

    for index, emotion in enumerate(emotion_names):

        value = float(
            emotions.get(
                emotion,
                0.0
            )
        )

        current_col = (
            emotion_col1
            if index % 2 == 0
            else emotion_col2
        )

        with current_col:

            st.markdown(
                f"""
                <div class="emotion-row">
                    <b>{emotion.capitalize()}</b>
                    — {value:.1f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    max(
                        value / 100,
                        0
                    ),
                    1
                )
            )

    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">📥 Analysis Report</div>',
        unsafe_allow_html=True
    )

    report = f"""
AI FACE ANALYSIS REPORT
=======================

Gender: {gender}
Gender Confidence: {gender_confidence:.1f}%

Estimated Age Range: {age_range}
Age Confidence: {age_confidence:.1f}%

Expression: {expression.capitalize()}
Expression Confidence: {expression_confidence:.1f}%

Face Detection Confidence: {detection_confidence:.1f}%

Expression Details
------------------
Angry: {float(emotions.get("angry", 0)):.1f}%
Disgust: {float(emotions.get("disgust", 0)):.1f}%
Fear: {float(emotions.get("fear", 0)):.1f}%
Happy: {float(emotions.get("happy", 0)):.1f}%
Sad: {float(emotions.get("sad", 0)):.1f}%
Surprise: {float(emotions.get("surprise", 0)):.1f}%
Neutral: {float(emotions.get("neutral", 0)):.1f}%

NOTE:
AI predictions are estimates.
Results may vary depending on lighting,
image quality, face angle and other factors.
"""

    st.download_button(
        label="⬇️ Download Analysis Report",
        data=report,
        file_name="AI_Face_Analysis_Report.txt",
        mime="text/plain",
        width="stretch"
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        "⚠️ AI predictions are estimates. "
        "Age, gender and expression results may vary "
        "depending on lighting, image quality, face angle "
        "and other factors."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🤖 AI Face Analysis System
        <br>
        <small>
        Powered by Computer Vision & AI
        </small>
    </div>
    """,
    unsafe_allow_html=True
)