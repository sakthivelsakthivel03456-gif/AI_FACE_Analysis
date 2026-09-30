import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="AI Face Analysis",
    page_icon="🤖"
)

st.title("🤖 AI Face Analysis")
st.write("Upload a photo or take a picture using your camera.")

# OpenCV built-in face detector
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

camera = st.camera_input("Take a picture")

uploaded = st.file_uploader(
    "Or upload an image",
    type=["jpg", "jpeg", "png"]
)

source = camera if camera is not None else uploaded

if source is not None:

    data = np.asarray(
        bytearray(source.getvalue()),
        dtype=np.uint8
    )

    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("Could not read the image.")
        st.stop()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    result = frame.copy()

    for (x, y, w, h) in faces:

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            "Face detected",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    result_rgb = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        result_rgb,
        caption="AI Face Analysis",
        use_container_width=True
    )

    st.subheader("Analysis Result")

    if len(faces) == 0:

        st.warning(
            "No clear face detected. "
            "Please try a photo with better lighting."
        )

    else:

        st.success(
            f"{len(faces)} face(s) detected."
        )

        # Optional user-provided information
        st.subheader("Personal Information")

        gender = st.selectbox(
            "Gender",
            [
                "Prefer not to say",
                "Male",
                "Female"
            ]
        )

        age_group = st.selectbox(
            "Age Group",
            [
                "Prefer not to say",
                "0-12",
                "13-17",
                "18-24",
                "25-32",
                "33-44",
                "45-60",
                "60+"
            ]
        )

        st.write("### Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Faces Detected",
                len(faces)
            )

        with col2:
            st.metric(
                "Image Status",
                "Clear"
            )

        if gender != "Prefer not to say":
            st.write(f"**Gender:** {gender}")

        if age_group != "Prefer not to say":
            st.write(f"**Age Group:** {age_group}")