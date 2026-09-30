import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace

# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="AI Face Analysis",
    page_icon="🤖"
)

st.title("🤖 AI Face Analysis")
st.write("Upload a photo or take a picture using your camera.")

# --------------------------------------------------
# FACE DETECTOR
# --------------------------------------------------

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# --------------------------------------------------
# INPUT
# --------------------------------------------------

camera = st.camera_input("Take a picture")

uploaded = st.file_uploader(
    "Or upload an image",
    type=["jpg", "jpeg", "png"]
)

source = camera if camera is not None else uploaded

# --------------------------------------------------
# PROCESS IMAGE
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
        st.error("Could not read image.")
        st.stop()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(100, 100)
    )

    # Keep largest face only
    if len(faces) > 0:
        faces = [
            max(
                faces,
                key=lambda f: f[2] * f[3]
            )
        ]

    result_frame = frame.copy()

    # --------------------------------------------------
    # NO FACE
    # --------------------------------------------------

    if len(faces) == 0:

        st.warning(
            "No clear face detected. "
            "Please try again with better lighting."
        )

        st.stop()

    # --------------------------------------------------
    # FACE FOUND
    # --------------------------------------------------

    x, y, w, h = faces[0]

    x = max(0, x)
    y = max(0, y)

    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)

    face = frame[y:y2, x:x2]

    if face.size == 0:
        st.error("Unable to process the detected face.")
        st.stop()

    # --------------------------------------------------
    # DRAW FACE BOX
    # --------------------------------------------------

    cv2.rectangle(
        result_frame,
        (x, y),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        result_frame,
        "Face detected",
        (x, max(30, y - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------
    # EXPRESSION ANALYSIS
    # --------------------------------------------------

    expression = "Unknown"
    expression_confidence = 0.0
    emotions = {}

    try:

        analysis = DeepFace.analyze(
            img_path=face,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
            silent=True
        )

        if isinstance(analysis, list):
            analysis = analysis[0]

        emotions = analysis.get("emotion", {})

        if emotions:

            detected_expression = max(
                emotions,
                key=emotions.get
            )

            expression = detected_expression.capitalize()

            expression_confidence = float(
                emotions[detected_expression]
            )

    except Exception as e:

        expression = "Unknown"
        expression_confidence = 0.0

    # --------------------------------------------------
    # WRITE EXPRESSION ON IMAGE
    # --------------------------------------------------

    cv2.putText(
        result_frame,
        "Expression: " + expression,
        (x, min(result_frame.shape[0] - 20, y2 + 35)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------

    result_rgb = cv2.cvtColor(
        result_frame,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        result_rgb,
        caption="AI Face Analysis",
        use_container_width=True
    )

    # --------------------------------------------------
    # USER DETAILS
    # --------------------------------------------------

    st.subheader("Personal Details")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Prefer not to say"
            ]
        )

    with col2:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=18,
            step=1
        )

    # --------------------------------------------------
    # ANALYSIS RESULT
    # --------------------------------------------------

    st.subheader("Analysis Result")

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

    st.write(
        f"**Expression Confidence:** "
        f"{expression_confidence:.1f}%"
    )

    # --------------------------------------------------
    # ALL EXPRESSIONS
    # --------------------------------------------------

    st.subheader("Expression Details")

    emotion_names = [
        "angry",
        "neutral",
        "happy",
        "sad",
        "fear",
        "disgust",
        "surprise"
    ]

    if emotions:

        for emotion in emotion_names:

            value = float(
                emotions.get(
                    emotion,
                    0.0
                )
            )

            st.write(
                f"**{emotion.capitalize()}:** "
                f"{value:.1f}%"
            )

            st.progress(
                min(
                    max(value / 100, 0.0),
                    1.0
                )
            )

    else:

        st.info(
            "Expression details are unavailable."
        )