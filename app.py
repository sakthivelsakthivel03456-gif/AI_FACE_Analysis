import os
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace


# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Face Analysis",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Face Analysis")
st.write("Upload a photo or take a picture using your camera.")


# =========================
# MODEL PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FACE_PROTO = str(MODELS_DIR / "opencv_face_detector.pbtxt")
FACE_MODEL = str(MODELS_DIR / "opencv_face_detector_uint8.pb")

AGE_PROTO = str(MODELS_DIR / "age_deploy.prototxt")
AGE_MODEL = str(MODELS_DIR / "age_net.caffemodel")

GENDER_PROTO = str(MODELS_DIR / "gender_deploy.prototxt")
GENDER_MODEL = str(MODELS_DIR / "gender_net.caffemodel")


# =========================
# LABELS
# =========================

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


# =========================
# CONFIDENCE SETTINGS
# =========================

FACE_CONFIDENCE = 0.75
GENDER_MIN_CONFIDENCE = 70.0
AGE_MIN_CONFIDENCE = 70.0


# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():

    required_files = [
        FACE_MODEL,
        FACE_PROTO,
        AGE_MODEL,
        AGE_PROTO,
        GENDER_MODEL,
        GENDER_PROTO
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            st.error(
                f"Model file missing:\n\n{file_path}"
            )

            st.stop()

    try:

        face_net = cv2.dnn.readNet(
            FACE_MODEL,
            FACE_PROTO
        )

        age_net = cv2.dnn.readNet(
            AGE_MODEL,
            AGE_PROTO
        )

        gender_net = cv2.dnn.readNet(
            GENDER_MODEL,
            GENDER_PROTO
        )

        return face_net, age_net, gender_net

    except Exception as e:

        st.error(
            "❌ Unable to load AI models."
        )

        st.code(str(e))

        st.stop()


face_net, age_net, gender_net = load_models()


# =========================
# IMAGE INPUT
# =========================

camera_image = st.camera_input(
    "Take a picture"
)

uploaded_image = st.file_uploader(
    "Or upload an image",
    type=["jpg", "jpeg", "png"]
)


source = (
    camera_image
    if camera_image is not None
    else uploaded_image
)


# =========================
# PROCESS IMAGE
# =========================

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


    height, width = frame.shape[:2]


    # =========================
    # FACE DETECTION
    # =========================

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


    detected_faces = []


    for i in range(detections.shape[2]):

        confidence = float(
            detections[0, 0, i, 2]
        )

        if confidence < FACE_CONFIDENCE:
            continue


        x1 = int(
            detections[0, 0, i, 3] * width
        )

        y1 = int(
            detections[0, 0, i, 4] * height
        )

        x2 = int(
            detections[0, 0, i, 5] * width
        )

        y2 = int(
            detections[0, 0, i, 6] * height
        )


        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)


        if x2 <= x1 or y2 <= y1:
            continue


        face = frame[
            y1:y2,
            x1:x2
        ]


        if face.size == 0:
            continue


        detected_faces.append(
            (x1, y1, x2, y2, face)
        )


    # =========================
    # NO FACE
    # =========================

    if len(detected_faces) == 0:

        st.warning(
            "⚠️ No clear face detected. "
            "Please try a clearer photo with better lighting."
        )

        st.stop()


    # =========================
    # PROCESS FIRST FACE
    # =========================

    x1, y1, x2, y2, face = detected_faces[0]


    # =========================
    # GENDER
    # =========================

    face_blob = cv2.dnn.blobFromImage(
        face,
        1.0,
        (227, 227),
        MODEL_MEAN_VALUES,
        swapRB=False
    )


    gender_net.setInput(face_blob)

    gender_prediction = gender_net.forward()[0]

    gender_index = int(
        np.argmax(gender_prediction)
    )

    gender_confidence = float(
        gender_prediction[gender_index] * 100
    )


    if gender_confidence >= GENDER_MIN_CONFIDENCE:

        gender = GENDER_LIST[gender_index]

    else:

        gender = "Uncertain"


    # =========================
    # AGE
    # =========================

    age_net.setInput(face_blob)

    age_prediction = age_net.forward()[0]

    age_index = int(
        np.argmax(age_prediction)
    )

    age_confidence = float(
        age_prediction[age_index] * 100
    )


    if age_confidence >= AGE_MIN_CONFIDENCE:

        age = AGE_LIST[age_index]

    else:

        age = "Uncertain"


    # =========================
    # EXPRESSION
    # =========================

    expression = "Unknown"
    expression_confidence = 0.0


    try:

        result = DeepFace.analyze(
            img_path=face,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
            silent=True
        )


        if isinstance(result, list):

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
                emotions[expression]
            )


    except Exception:

        expression = "Unknown"
        expression_confidence = 0.0


    # =========================
    # DRAW FACE BOX
    # =========================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # =========================
    # DISPLAY LABELS
    # =========================

    cv2.putText(
        frame,
        "Gender: " + gender,
        (x1, max(30, y1 - 65)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "Age: " + age,
        (x1, max(55, y1 - 35)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "Expression: " + expression,
        (x1, min(height - 10, y2 + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    # =========================
    # SHOW IMAGE
    # =========================

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    st.image(
        frame_rgb,
        caption="AI Face Analysis",
        use_container_width=True
    )


    # =========================
    # RESULTS
    # =========================

    st.subheader(
        "Analysis Result"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Gender",
            gender
        )


    with col2:

        st.metric(
            "Age",
            age
        )


    with col3:

        st.metric(
            "Expression",
            expression
        )


    # =========================
    # CONFIDENCE
    # =========================

    st.write(
        f"**Gender Confidence:** "
        f"{gender_confidence:.1f}%"
    )


    st.write(
        f"**Age Confidence:** "
        f"{age_confidence:.1f}%"
    )


    if expression != "Unknown":

        st.write(
            f"**Expression Confidence:** "
            f"{expression_confidence:.1f}%"
        )

    else:

        st.write(
            "**Expression Confidence:** N/A"
        )


    # =========================
    # INFORMATION
    # =========================

    st.info(
        "Note: Age, gender and expression predictions "
        "are AI estimates and may not always be accurate."
    )