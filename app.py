
import os
import urllib.request

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace

# =====================================================
# AI FACE ANALYSIS - TEAM NIGHT FURRY
# OpenCV: face detection, age and gender-category estimates
# DeepFace: expression estimate
# =====================================================

st.set_page_config(
    page_title="AI Face Analysis",
    page_icon="🤖",
    layout="wide",
)

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

GENDER_LIST = [
    "Male-category estimate",
    "Female-category estimate",
]

MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746,
)


# =====================================================
# DOWNLOAD MODEL FILES
# =====================================================

def download_model(filename, url):
    path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path

    temp_path = path + ".download"

    try:
        with st.spinner(f"Preparing model: {filename}..."):
            urllib.request.urlretrieve(url, temp_path)

        if os.path.getsize(temp_path) <= 1000:
            raise RuntimeError(
                "Downloaded model file is unexpectedly small."
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


# =====================================================
# LOAD OPENCV MODELS
# =====================================================

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


# =====================================================
# FACE DETECTION
# =====================================================

def detect_faces(frame, face_net, threshold=0.50):
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

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 > x1 and y2 > y1:
            faces.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
            })

    # Largest face first. Only the largest is analyzed later.
    faces.sort(
        key=lambda face:
            (face["x2"] - face["x1"])
            * (face["y2"] - face["y1"]),
        reverse=True,
    )

    return faces


# =====================================================
# PADDED SQUARE CROP FOR EXPRESSION ANALYSIS
# =====================================================

def square_face_crop(
    frame,
    x1,
    y1,
    x2,
    y2,
    padding_ratio=0.15,
):
    height, width = frame.shape[:2]

    face_width = x2 - x1
    face_height = y2 - y1

    side = max(
        32,
        int(
            max(face_width, face_height)
            * (1.0 + 2.0 * padding_ratio)
        ),
    )

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    left = center_x - side // 2
    top = center_y - side // 2
    right = left + side
    bottom = top + side

    crop_left = max(0, left)
    crop_top = max(0, top)
    crop_right = min(width, right)
    crop_bottom = min(height, bottom)

    crop = frame[
        crop_top:crop_bottom,
        crop_left:crop_right,
    ]

    if crop.size == 0:
        return frame[y1:y2, x1:x2].copy()

    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - height)
    pad_right = max(0, right - width)

    if pad_top or pad_bottom or pad_left or pad_right:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE,
        )

    return crop


# =====================================================
# AGE AND GENDER-CATEGORY ESTIMATION
# =====================================================

def predict_age_gender(face, age_net, gender_net):
    if face is None or face.size == 0:
        return "Unknown", 0.0, "Unavailable", 0.0

    # Use the tight, detected face crop for these Caffe models.
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
        gender_scores[gender_index] * 100.0
    )

    age_net.setInput(blob)
    age_scores = age_net.forward()[0]

    age_index = int(np.argmax(age_scores))
    age_range = AGE_LIST[age_index]

    age_score = float(
        age_scores[age_index] * 100.0
    )

    return (
        gender,
        gender_score,
        age_range,
        age_score,
    )


# =====================================================
# EXPRESSION ESTIMATION
# =====================================================

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

        dominant = str(
            result.get("dominant_emotion", "Unknown")
        )

        score = (
            float(emotions.get(dominant, 0.0))
            if dominant != "Unknown"
            else 0.0
        )

        return dominant, score, emotions

    except Exception as exc:
        return "Unknown", 0.0, {
            "analysis_error": str(exc)
        }


# =====================================================
# APPLICATION UI
# =====================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        🤖 AI Face Analysis System
    </h1>
    <p style="text-align:center;color:#999;">
        Face Detection • Approximate Age • Gender Category • Expression
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")

    input_method = st.radio(
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
        "AI results are estimates. Model scores do not represent "
        "real-world prediction accuracy."
    )


if input_method == "📷 Camera":
    source = st.camera_input("Take a picture")
else:
    source = st.file_uploader(
        "Upload a clear face photo",
        type=["jpg", "jpeg", "png"],
    )


# =====================================================
# PROCESS IMAGE AND ANALYZE ONE FACE
# =====================================================

if source is not None:
    image_array = np.frombuffer(
        source.getvalue(),
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        st.error("Image read panna mudiyala da. JPG/PNG try pannu.")
        st.stop()

    try:
        with st.spinner("Loading face, age and gender models..."):
            face_net, age_net, gender_net = load_models()

    except Exception as exc:
        st.error("OpenCV models load aagala.")
        st.exception(exc)
        st.stop()

    with st.spinner("Finding the largest face..."):
        faces = detect_faces(frame, face_net)

    if not faces:
        st.warning(
            "Face detect panna mudiyala da. "
            "Clear, front-facing photo try pannu."
        )
        st.stop()

    # Select the largest face only.
    selected_face = faces[0]

    x1 = selected_face["x1"]
    y1 = selected_face["y1"]
    x2 = selected_face["x2"]
    y2 = selected_face["y2"]

    # Tight crop for age/gender estimation.
    age_gender_crop = frame[y1:y2, x1:x2].copy()

    if age_gender_crop.size == 0:
        st.error("Face crop empty-ah irukku da. Vera photo try pannu.")
        st.stop()

    # Padded crop for expression estimation.
    emotion_crop = square_face_crop(
        frame,
        x1,
        y1,
        x2,
        y2,
    )

    with st.spinner("Analyzing the selected face..."):
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

        expression, expression_score, emotions = predict_emotion(
            emotion_crop
        )

    # Draw one rectangle only.
    marked = frame.copy()

    cv2.rectangle(
        marked,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    gender_short = gender.split("-")[0]

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

    cv2.rectangle(
        marked,
        (x1, text_y - text_height - 8),
        (
            min(
                frame.shape[1] - 1,
                x1 + text_width + 10,
            ),
            text_y + baseline,
        ),
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

    # Show marked image.
    st.subheader("🔍 Face Detection Result")

    st.image(
        cv2.cvtColor(marked, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )

    st.caption(
        f"Faces detected: {len(faces)}. "
        "Only the largest detected face is analyzed."
    )

    # -------------------------------------------------
    # ANALYSIS RESULT
    # -------------------------------------------------

    st.subheader("📊 Analysis Result — Main Face Only")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Gender category (estimate)",
        gender,
    )

    col2.metric(
        "Estimated age range",
        age_range,
    )

    col3.metric(
        "Expression estimate",
        expression.capitalize(),
    )

    # -------------------------------------------------
    # MODEL SCORES
    # -------------------------------------------------

    st.subheader("Model scores (not accuracy)")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Gender model score",
        f"{gender_score:.1f}%",
    )

    col2.metric(
        "Age model score",
        f"{age_score:.1f}%",
    )

    col3.metric(
        "Expression model score",
        f"{expression_score:.1f}%",
    )

    st.caption(
        f"Face detector score: "
        f"{selected_face['confidence'] * 100:.1f}% "
        "(not prediction accuracy)."
    )

    # -------------------------------------------------
    # DEBUG CROPS
    # -------------------------------------------------

    with st.expander("View crops sent to the models"):
        crop_col1, crop_col2 = st.columns(2)

        with crop_col1:
            st.caption("Age/Gender model input (tight crop)")

            st.image(
                cv2.cvtColor(
                    age_gender_crop,
                    cv2.COLOR_BGR2RGB,
                ),
                width=220,
            )

        with crop_col2:
            st.caption("Expression model input (padded crop)")

            st.image(
                cv2.cvtColor(
                    emotion_crop,
                    cv2.COLOR_BGR2RGB,
                ),
                width=220,
            )

    # -------------------------------------------------
    # EXPRESSION SCORES
    # -------------------------------------------------

    if emotions and "analysis_error" not in emotions:
        st.subheader("😊 Expression Details")

        for name in (
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ):
            if name not in emotions:
                continue

            value = max(
                0.0,
                min(100.0, float(emotions[name])),
            )

            st.write(
                f"{name.capitalize()}: "
                f"{value:.1f}% model score"
            )

            st.progress(int(round(value)))

    elif "analysis_error" in emotions:
        st.warning(
            "Emotion analysis unavailable: "
            + emotions["analysis_error"]
        )

    st.info(
        "Age, gender-category and expression are AI estimates "
        "and may be incorrect. Model scores are not real-world "
        "accuracy rates."
    )

else:
    st.markdown(
        """
        ### 👋 Welcome!

        Upload an image or use the camera to start face analysis.
        """
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <p style="text-align:center;color:#888;margin-top:35px;">
        🤖 AI Face Analysis System | Team Night Furry
    </p>
    """,
    unsafe_allow_html=True,
)