import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Face Analysis",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #888888;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 20px;
        border-radius: 15px;
        background: #f5f5f5;
        text-align: center;
        margin-bottom: 10px;
    }

    .result-title {
        font-size: 15px;
        color: #777777;
        margin-bottom: 5px;
    }

    .result-value {
        font-size: 26px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #888888;
        font-size: 14px;
        margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🤖 AI Face Analysis System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI-powered face, age, gender and expression analysis'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("Choose an image source:")

    source_type = st.radio(
        "Input Method",
        [
            "📷 Camera",
            "🖼️ Upload Image"
        ]
    )

    st.divider()

    st.info(
        "AI predictions are estimates. "
        "Results can vary depending on lighting, "
        "image quality and face angle."
    )


# =========================================================
# INPUT
# =========================================================

image_source = None


if source_type == "📷 Camera":

    image_source = st.camera_input(
        "Take a picture"
    )

else:

    image_source = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


# =========================================================
# START ANALYSIS
# =========================================================

if image_source is not None:

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    image_bytes = image_source.getvalue()

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
            "❌ Unable to read the image."
        )

        st.stop()


    # -----------------------------------------------------
    # ORIGINAL IMAGE
    # -----------------------------------------------------

    st.subheader("🖼️ Input Image")

    image_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        image_rgb,
        width="stretch"
    )


    # =====================================================
    # DEEPFACE ANALYSIS
    # =====================================================

    with st.spinner(
        "🤖 AI is analyzing the face..."
    ):

        try:

            analysis = DeepFace.analyze(
                img_path=frame,
                actions=[
                    "age",
                    "gender",
                    "emotion"
                ],
                detector_backend="retinaface",
                enforce_detection=True,
                align=True
            )

        except Exception as e:

            st.error(
                "❌ Face analysis failed."
            )

            st.warning(
                "Please upload a clear face image "
                "with good lighting and the complete "
                "face visible."
            )

            st.stop()


    # =====================================================
    # MULTIPLE FACE HANDLING
    # =====================================================

    if isinstance(analysis, list):

        # Select the largest detected face
        analysis = max(
            analysis,
            key=lambda item: (
                item.get("region", {}).get("w", 0)
                *
                item.get("region", {}).get("h", 0)
            )
        )


    # =====================================================
    # AGE
    # =====================================================

    age = analysis.get(
        "age",
        "Unknown"
    )


    # =====================================================
    # GENDER
    # =====================================================

    gender_data = analysis.get(
        "gender",
        {}
    )

    gender = analysis.get(
        "dominant_gender",
        "Unknown"
    )

    gender_confidence = 0.0


    if isinstance(gender_data, dict):

        if gender == "Man":

            gender_confidence = float(
                gender_data.get(
                    "Man",
                    0
                )
            )

        elif gender == "Woman":

            gender_confidence = float(
                gender_data.get(
                    "Woman",
                    0
                )
            )


    # =====================================================
    # EMOTION
    # =====================================================

    emotions = analysis.get(
        "emotion",
        {}
    )

    expression = analysis.get(
        "dominant_emotion",
        "Unknown"
    )

    expression_confidence = 0.0


    if isinstance(emotions, dict):

        expression_confidence = float(
            emotions.get(
                expression,
                0
            )
        )


    # =====================================================
    # FACE REGION
    # =====================================================

    result_frame = frame.copy()

    region = analysis.get(
        "region",
        {}
    )


    if isinstance(region, dict):

        x = int(
            region.get(
                "x",
                0
            )
        )

        y = int(
            region.get(
                "y",
                0
            )
        )

        w = int(
            region.get(
                "w",
                0
            )
        )

        h = int(
            region.get(
                "h",
                0
            )
        )


        # Make sure coordinates stay inside image

        x = max(
            0,
            x
        )

        y = max(
            0,
            y
        )

        x2 = min(
            frame.shape[1],
            x + w
        )

        y2 = min(
            frame.shape[0],
            y + h
        )


        if x2 > x and y2 > y:

            cv2.rectangle(
                result_frame,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                3
            )


    # =====================================================
    # DETECTED FACE IMAGE
    # =====================================================

    result_rgb = cv2.cvtColor(
        result_frame,
        cv2.COLOR_BGR2RGB
    )

    st.subheader("🔍 Detected Face")

    st.image(
        result_rgb,
        width="stretch"
    )


    # =====================================================
    # PERSONAL DETAILS
    # =====================================================

    st.subheader(
        "👤 Personal Details"
    )

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    Detected Gender
                </div>
                <div class="result-value">
                    {gender}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    Estimated Age
                </div>
                <div class="result-value">
                    {age}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # ANALYSIS RESULT
    # =====================================================

    st.subheader(
        "📊 Analysis Result"
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
            str(age)
        )


    with col3:

        st.metric(
            "Expression",
            expression.capitalize()
        )


    # =====================================================
    # MODEL CONFIDENCE
    # =====================================================

    st.subheader(
        "🎯 Model Confidence"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.write(
            f"**Gender:** "
            f"{gender_confidence:.1f}%"
        )

        st.progress(
            min(
                max(
                    gender_confidence / 100,
                    0.0
                ),
                1.0
            )
        )


    with col2:

        st.write(
            "**Age:** Estimated"
        )

        st.progress(
            0.0
        )


    with col3:

        st.write(
            f"**Expression:** "
            f"{expression_confidence:.1f}%"
        )

        st.progress(
            min(
                max(
                    expression_confidence / 100,
                    0.0
                ),
                1.0
            )
        )


    # =====================================================
    # EXPRESSION DETAILS
    # =====================================================

    st.subheader(
        "😊 Expression Details"
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


    if isinstance(
        emotions,
        dict
    ):

        for emotion in emotion_names:

            value = float(
                emotions.get(
                    emotion,
                    0.0
                )
            )

            st.write(
                f"**{emotion.capitalize()}** "
                f"— {value:.1f}%"
            )

            st.progress(
                min(
                    max(
                        value / 100,
                        0.0
                    ),
                    1.0
                )
            )

    else:

        st.info(
            "Expression details are unavailable."
        )


    # =====================================================
    # NOTE
    # =====================================================

    st.info(
        "ℹ️ Note: AI predictions are estimates. "
        "Age, gender and expression results can vary "
        "depending on lighting, image quality and "
        "face angle."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🤖 AI Face Analysis System
    </div>
    """,
    unsafe_allow_html=True
)