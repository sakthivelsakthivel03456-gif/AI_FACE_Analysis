import cv2
import numpy as np
from collections import Counter, deque

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None


# ============================================================
# MODEL FILE PATHS
# ============================================================

faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"


# ============================================================
# LOAD AI MODELS
# ============================================================

print("Loading AI models...")

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

print("AI models loaded successfully!")
if DeepFace is None:
    print("WARNING: DeepFace is not installed. Expression detection will be unavailable.")


# ============================================================
# LABELS
# ============================================================

ageList = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]

genderList = [
    "Male",
    "Female"
]

MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print()
print("======================================")
print(" AI FACE ANALYSIS SYSTEM STARTED")
print("======================================")
print("Press Q to close camera.")
print()


# ============================================================
# AGE / GENDER SMOOTHING
# ============================================================

age_history = deque(maxlen=8)
gender_history = deque(maxlen=8)
emotion_history = deque(maxlen=8)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    height, width = frame.shape[:2]


    # ========================================================
    # LIGHTING ANALYSIS
    # ========================================================

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray_frame)

    # Default blur value
    blur_value = 100


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

    faceNet.setInput(blob)

    detections = faceNet.forward()

    face_found = False


    # ========================================================
    # PROCESS FACES
    # ========================================================

    for i in range(detections.shape[2]):

        confidence = float(detections[0, 0, i, 2])

        # Higher confidence = fewer wrong face detections
        if confidence < 0.75:
            continue


        # ----------------------------------------------------
        # FACE COORDINATES
        # ----------------------------------------------------

        x1 = int(detections[0, 0, i, 3] * width)
        y1 = int(detections[0, 0, i, 4] * height)
        x2 = int(detections[0, 0, i, 5] * width)
        y2 = int(detections[0, 0, i, 6] * height)


        # Keep inside frame

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)


        # Check valid face

        if x2 <= x1 or y2 <= y1:
            continue


        face_width = x2 - x1
        face_height = y2 - y1


        # Ignore very small faces

        if face_width < 80 or face_height < 80:
            continue


        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue


        face_found = True


        # ====================================================
        # FACE BLUR DETECTION
        # ====================================================

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        blur_value = cv2.Laplacian(
            face_gray,
            cv2.CV_64F
        ).var()


        # ====================================================
        # FACE IMAGE PREPARATION
        # ====================================================

        faceBlob = cv2.dnn.blobFromImage(
            face,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )


        # ====================================================
        # GENDER PREDICTION
        # ====================================================

        genderNet.setInput(faceBlob)

        genderPred = genderNet.forward()

        genderIndex = int(np.argmax(genderPred[0]))

        genderConfidence = float(
            genderPred[0][genderIndex]
        )

        gender = genderList[genderIndex]


        # ====================================================
        # AGE PREDICTION
        # ====================================================

        ageNet.setInput(faceBlob)

        agePred = ageNet.forward()

        ageIndex = int(np.argmax(agePred[0]))

        ageConfidence = float(
            agePred[0][ageIndex]
        )

        age = ageList[ageIndex]

        # ====================================================
        # EXPRESSION / EMOTION PREDICTION
        # ====================================================

        expression = "Unknown"
        expressionConfidence = 0.0

        if DeepFace is not None:
            try:
                emotion_result = DeepFace.analyze(
                    img_path=face,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="skip",
                    silent=True
                )

                if isinstance(emotion_result, list):
                    emotion_result = emotion_result[0]

                emotion_scores = emotion_result.get("emotion", {})
                if emotion_scores:
                    expression = max(
                        emotion_scores,
                        key=emotion_scores.get
                    )
                    expressionConfidence = float(
                        emotion_scores[expression]
                    ) / 100.0

            except Exception as e:
                expression = "Unknown"
                expressionConfidence = 0.0

        # ====================================================
        # SMOOTH AGE PREDICTION
        # ====================================================

        age_history.append(age)

        stable_age = Counter(
            age_history
        ).most_common(1)[0][0]


        # ====================================================
        # SMOOTH GENDER PREDICTION
        # ====================================================

        gender_history.append(gender)

        stable_gender = Counter(
            gender_history
        ).most_common(1)[0][0]

        if expression != "Unknown":
            emotion_history.append(expression)

        if emotion_history:
            stable_expression = Counter(
                emotion_history
            ).most_common(1)[0][0]
        else:
            stable_expression = "Unknown"


        # ====================================================
        # FACE BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ====================================================
        # DISPLAY AGE + GENDER
        # ====================================================

        text1 = "Gender: " + stable_gender
        text2 = "Age: " + stable_age
        text3 = "Expression: " + stable_expression


        cv2.putText(
            frame,
            text1,
            (x1, max(25, y1 - 45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            text2,
            (x1, max(50, y1 - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text3,
            (x1, max(75, y1 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        # ====================================================
        # CONFIDENCE DISPLAY
        # ====================================================

        ageConfText = (
            "Age Confidence: "
            + str(int(ageConfidence * 100))
            + "%"
        )

        cv2.putText(
            frame,
            ageConfText,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

        if expression != "Unknown":
            expressionConfText = (
                "Expression Confidence: "
                + str(int(expressionConfidence * 100))
                + "%"
            )
            cv2.putText(
                frame,
                expressionConfText,
                (x1, y2 + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )


        # ====================================================
        # LOW CONFIDENCE WARNING
        # ====================================================

        if ageConfidence < 0.45:

            cv2.putText(
                frame,
                "WARNING: Age prediction uncertain",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )


        # ====================================================
        # BLUR WARNING
        # ====================================================

        if blur_value < 25:

            cv2.putText(
                frame,
                "WARNING: IMAGE BLUR - Keep camera steady",
                (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )


    # ========================================================
    # LIGHTING WARNINGS
    # ========================================================

    if brightness < 55:

        cv2.putText(
            frame,
            "WARNING: LOW LIGHT - Improve lighting",
            (20, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    elif brightness > 210:

        cv2.putText(
            frame,
            "WARNING: TOO BRIGHT - Reduce light",
            (20, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


    # ========================================================
    # FACE NOT FOUND
    # ========================================================

    if not face_found:

        cv2.putText(
            frame,
            "No clear face detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================

    info = "Brightness: " + str(int(brightness))

    cv2.putText(
        frame,
        info,
        (20, height - 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "AI Face Analysis System",
        frame
    )


    # ========================================================
    # PRESS Q TO EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLOSE
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("AI Face Analysis Closed.")