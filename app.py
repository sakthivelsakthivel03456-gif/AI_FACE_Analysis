import os
import sys
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace


# ============================================================
# UTF-8 OUTPUT
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="AI Face Analysis",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 18px;
        border-radius: 15px;
        background: rgba(50, 100, 180, 0.15);
        border: 1px solid rgba(100, 150, 255, 0.25);
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .result-card {
        padding: 22px;
        border-radius: 18px;
        text-align: center;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.10);
        min-height: 130px;
    }

    .result-title {
        font-size: 17px;
        opacity: 0.75;
    }

    .result-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 10px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        opacity: 0.55;
        margin-top: 40px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Face Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Face, Age, Gender & Expression Analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FACE_PROTO = str(
    MODELS_DIR / "opencv_face_detector.pbtxt"
)

FACE_MODEL = str(
    MODELS_DIR / "opencv_face_detector_uint8.pb"
)

AGE_PROTO = str(
    MODELS_DIR / "age_deploy.prototxt"
)

AGE_MODEL = str(
    MODELS_DIR / "age_net.caffemodel"
)

GENDER_PROTO = str(
    MODELS_DIR / "gender_deploy.prototxt"
)

GENDER_MODEL = str(
    MODELS_DIR / "gender_net.caffemodel"
)


# ============================================================
# AGE / GENDER LABELS
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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        "Upload an image or use your camera "
        "to perform AI face analysis."
    )

    st.divider()

    st.subheader("🔍 Detection")

    st.write("• Face Detection")
    st.write("• Age Estimation")
    st.write("• Gender Estimation")
    st.write("• Expression Detection")

    st.divider()

    st.info(
        "AI predictions are estimates and may vary "
        "depending on image quality, lighting and face angle."
    )


# ============================================================
# LOAD OPENCV MODELS
# ============================================================

@st.cache_resource
def load_models():

    required_files = [
        (FACE_MODEL, "opencv_face_detector_uint8.pb"),
        (FACE_PROTO, "opencv_face_detector.pbtxt"),
        (AGE_MODEL, "age_net.caffemodel"),
        (AGE_PROTO, "age_deploy.prototxt"),
        (GENDER_MODEL, "gender_net.caffemodel"),
        (GENDER_PROTO, "gender_deploy.prototxt"),
    ]

    missing = []

    for path, name in required_files:

        if not os.path.isfile(path):
            missing.append(name)

    if missing:

        st.error(
            "❌ Missing model file(s): "
            + ", ".join(missing)
        )

        st.stop()

    try:

        face_net = cv2.dnn.readNetFromTensorflow(
            FACE_MODEL,
            FACE_PROTO
        )

        age_net = cv2.dnn.readNetFromCaffe(
            AGE_PROTO,
            AGE_MODEL
        )

        gender_net = cv2.dnn.readNetFromCaffe(
            GENDER_PROTO,
            GENDER_MODEL
        )

        return (
            face_net,
            age_net,
            gender_net
        )

    except Exception as e:

        st.error("❌ Model loading failed")
        st.code(str(e))

        st.stop()


# Load models
face_net, age_net, gender_net = load_models()


# ============================================================
# IMAGE INPUT
# ============================================================

col_camera, col_upload = st.columns(2)


with col_camera:

    st.subheader("📷 Take a picture")

    camera = st.camera_input(
        "Camera",
        label_visibility="collapsed"
    )


with col_upload:

    st.subheader("📁 Upload an image")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


# ============================================================
# SELECT IMAGE
# ============================================================

source = None

if camera is not None:

    source = camera

elif uploaded is not None:

    source = uploaded


# ============================================================
# PROCESS IMAGE
# ============================================================

if source is not None:

    data = np.asarray(
        bytearray(source.getvalue()),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        st.error(
            "❌ Could not read image. "
            "Please select a valid JPG or PNG image."
        )

        st.stop()


    height, width = frame.shape[:2]


    # ========================================================
    # FACE DETECTION
    # ========================================================

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

    face_found = False


    # Default values

    gender = "Unknown"
    age = "Unknown"
    expression = "Unknown"

    gender_confidence = 0.0
    age_confidence = 0.0
    expression_confidence = 0.0


    # ========================================================
    # PROCESS FIRST DETECTED FACE
    # ========================================================

    for i in range(detections.shape[2]):

        confidence = float(
            detections[0, 0, i, 2]
        )

        if confidence < 0.75:
            continue


        # ----------------------------------------------------
        # FACE COORDINATES
        # ----------------------------------------------------

        x1 = int(
            detections[0, 0, i, 3]
            * width
        )

        y1 = int(
            detections[0, 0, i, 4]
            * height
        )

        x2 = int(
            detections[0, 0, i, 5]
            * width
        )

        y2 = int(
            detections[0, 0, i, 6]
            * height
        )


        # Keep inside image

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)


        # ----------------------------------------------------
        # FACE SIZE CHECK
        # ----------------------------------------------------

        if x2 <= x1 or y2 <= y1:
            continue

        face = frame[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:
            continue

        face_found = True


        # ====================================================
        # FACE BLOB
        # ====================================================

        face_blob = cv2.dnn.blobFromImage(
            face,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )


        # ====================================================
        # GENDER PREDICTION
        # ====================================================

        gender_net.setInput(
            face_blob
        )

        gender_prediction = (
            gender_net.forward()[0]
        )

        gender_index = int(
            np.argmax(
                gender_prediction
            )
        )

        gender = GENDER_LIST[
            gender_index
        ]

        gender_confidence = float(
            gender_prediction[
                gender_index
            ] * 100
        )


        # ====================================================
        # AGE PREDICTION
        # ====================================================

        age_net.setInput(
            face_blob
        )

        age_prediction = (
            age_net.forward()[0]
        )

        age_index = int(
            np.argmax(
                age_prediction
            )
        )

        age = AGE_LIST[
            age_index
        ]

        age_confidence = float(
            age_prediction[
                age_index
            ] * 100
        )


        # ====================================================
        # EXPRESSION PREDICTION
        # ====================================================

        try:

            result = DeepFace.analyze(
                img_path=face,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True
            )

            if isinstance(
                result,
                list
            ):

                result = result[0]


            emotions = result.get(
                "emotion",
                {}
            )


            if emotions:

                expression = max(
                    emotions,
                    key=emotions.get
                )

                expression_confidence = float(
                    emotions[
                        expression
                    ]
                )


        except Exception:

            expression = "Unknown"

            expression_confidence = 0.0


        # ====================================================
        # DRAW FACE BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        # Gender

        cv2.putText(
            frame,
            "Gender: " + gender,
            (
                x1,
                max(30, y1 - 65)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Age

        cv2.putText(
            frame,
            "Age: " + age,
            (
                x1,
                max(55, y1 - 35)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Expression

        cv2.putText(
            frame,
            "Expression: " + expression,
            (
                x1,
                min(
                    height - 15,
                    y2 + 30
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Only first face

        break


    # ========================================================
    # SHOW RESULT
    # ========================================================

    if face_found:

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        st.markdown(
            '<div class="section-title">'
            '📊 Analysis Result'
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # RESULT CARDS
        # ----------------------------------------------------

        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">👤 Gender</div>
                    <div class="result-value">{gender}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with result_col2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">🎂 Age</div>
                    <div class="result-value">{age}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with result_col3:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">😊 Expression</div>
                    <div class="result-value">{expression}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # DETECTED IMAGE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🖼️ Detected Face'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            frame_rgb,
            caption="AI Face Analysis Result",
            width="stretch"
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🎯 Model Confidence'
            '</div>',
            unsafe_allow_html=True
        )


        conf1, conf2, conf3 = st.columns(3)


        with conf1:

            st.write(
                f"**Gender:** "
                f"{gender_confidence:.1f}%"
            )

            st.progress(
                min(
                    gender_confidence / 100,
                    1.0
                )
            )


        with conf2:

            st.write(
                f"**Age:** "
                f"{age_confidence:.1f}%"
            )

            st.progress(
                min(
                    age_confidence / 100,
                    1.0
                )
            )


        with conf3:

            st.write(
                f"**Expression:** "
                f"{expression_confidence:.1f}%"
            )

            st.progress(
                min(
                    expression_confidence / 100,
                    1.0
                )
            )


        # ----------------------------------------------------
        # EXPRESSION DETAILS
        # ----------------------------------------------------

        if expression != "Unknown":

            st.markdown(
                '<div class="section-title">'
                '😊 Expression Details'
                '</div>',
                unsafe_allow_html=True
            )


            try:

                emotion_result = DeepFace.analyze(
                    img_path=face,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="skip",
                    silent=True
                )


                if isinstance(
                    emotion_result,
                    list
                ):

                    emotion_result = emotion_result[0]


                emotion_scores = (
                    emotion_result.get(
                        "emotion",
                        {}
                    )
                )


                for emotion_name, score in emotion_scores.items():

                    st.write(
                        f"**{emotion_name.title()}** — "
                        f"{float(score):.1f}%"
                    )

                    st.progress(
                        min(
                            float(score) / 100,
                            1.0
                        )
                    )


            except Exception:

                st.info(
                    "Expression details could not be loaded."
                )


        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="info-box">
            ℹ️ <b>Note:</b> AI predictions are estimates.
            Age, gender and expression results can vary
            depending on lighting, image quality and face angle.
            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.warning(
            "⚠️ No clear face detected. "
            "Please try again with better lighting "
            "and keep your face clearly visible."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🤖 AI Face Analysis System<br>
        Powered by OpenCV + DeepFace
    </div>
    """,
    unsafe_allow_html=True
)