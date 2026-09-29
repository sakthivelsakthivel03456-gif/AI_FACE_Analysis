import os
import sys
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace


# UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Streamlit page
st.set_page_config(
    page_title="AI Face Analysis",
    page_icon="🤖"
)

st.title("🤖 AI Face Analysis")
st.write("Upload a photo or take a picture using your camera.")


# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

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


# --------------------------------------------------
# AGE / GENDER SETTINGS
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

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

    # Check model files
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

        # Face detection model
        face_net = cv2.dnn.readNetFromTensorflow(
            FACE_MODEL,
            FACE_PROTO
        )

        # Age model
        age_net = cv2.dnn.readNetFromCaffe(
            AGE_PROTO,
            AGE_MODEL
        )

        # Gender model
        gender_net = cv2.dnn.readNetFromCaffe(
            GENDER_PROTO,
            GENDER_MODEL
        )

        return face_net, age_net, gender_net

    except Exception as e:

        st.error("❌ Model loading failed")
        st.code(str(e))
        st.stop()


# Load models
face_net, age_net, gender_net = load_models()


# --------------------------------------------------
# CAMERA / IMAGE UPLOAD
# --------------------------------------------------

camera = st.camera_input(
    "Take a picture"
)

uploaded = st.file_uploader(
    "Or upload an image",
    type=["jpg", "jpeg", "png"]
)


# Select image source
source = (
    camera
    if camera is not None
    else uploaded
)


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

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
            "❌ Could not read image file. "
            "Please upload a valid image."
        )

        st.stop()


    height, width = frame.shape[:2]


    # --------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------

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


    # --------------------------------------------------
    # PROCESS DETECTED FACE
    # --------------------------------------------------

    for i in range(
        detections.shape[2]
    ):

        confidence = detections[
            0, 0, i, 2
        ]


        if confidence < 0.75:
            continue


        # Face coordinates
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


        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)


        # Crop face
        face = frame[
            y1:y2,
            x1:x2
        ]


        if face.size == 0:
            continue


        face_found = True


        # --------------------------------------------------
        # FACE BLOB
        # --------------------------------------------------

        face_blob = cv2.dnn.blobFromImage(
            face,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )


        # --------------------------------------------------
        # GENDER
        # --------------------------------------------------

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


        # --------------------------------------------------
        # AGE
        # --------------------------------------------------

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


        # --------------------------------------------------
        # EXPRESSION
        # --------------------------------------------------

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


        # --------------------------------------------------
        # DRAW FACE BOX
        # --------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Gender text
        cv2.putText(
            frame,
            "Gender: " + gender,
            (
                x1,
                max(
                    30,
                    y1 - 60
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Age text
        cv2.putText(
            frame,
            "Age: " + age,
            (
                x1,
                max(
                    55,
                    y1 - 30
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Expression text
        cv2.putText(
            frame,
            "Expression: "
            + expression,
            (
                x1,
                min(
                    height - 10,
                    y2 + 30
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Only process first face
        break


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    if face_found:

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        st.image(
            frame_rgb,
            caption="AI Face Analysis",
            use_container_width=True
        )


        st.subheader(
            "Analysis Result"
        )


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Gender",
            gender
        )


        col2.metric(
            "Age",
            age
        )


        col3.metric(
            "Expression",
            expression
        )


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


    else:

        st.warning(
            "⚠️ No clear face detected. "
            "Please try again with better lighting."
        )