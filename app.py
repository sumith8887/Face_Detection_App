import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras_facenet import FaceNet
from mtcnn import MTCNN

# ---------------- Load FaceNet ----------------
embedder = FaceNet()

# Optional: load your saved weights
embedder.model.load_weights("models/facenet.weights.h5")

# Face detector
detector = MTCNN()

# ---------------- Streamlit UI ----------------
st.set_page_config(
    page_title="Face Detection in Crowd",
    page_icon="🧑",
    layout="wide"
)

st.title("🧑 Face Detection in Crowd Image")
st.write(
    "Upload a person's face and a crowd image. "
    "The matching person will be highlighted."
)

# Upload files
target_file = st.file_uploader(
    "Upload Target Person Image",
    type=["jpg", "jpeg", "png"]
)

crowd_file = st.file_uploader(
    "Upload Crowd Image",
    type=["jpg", "jpeg", "png"]
)


# ---------------- Face Embedding Function ----------------
def get_embedding(face):

    face = cv2.resize(face, (160, 160))

    face = np.expand_dims(face, axis=0)

    embedding = embedder.embeddings(face)

    return embedding[0]


# ---------------- Main Logic ----------------
if target_file and crowd_file:

    # Target Image
    target_img = Image.open(target_file)
    target_img = np.array(target_img)

    target_faces = detector.detect_faces(target_img)

    if len(target_faces) == 0:

        st.error("No face detected in target image!")

    else:

        # First face from target image
        x, y, w, h = target_faces[0]["box"]

        x = abs(x)
        y = abs(y)

        target_face = target_img[y:y+h, x:x+w]

        target_embedding = get_embedding(target_face)

        # Crowd Image
        crowd_img = Image.open(crowd_file)
        crowd_img = np.array(crowd_img)

        crowd_faces = detector.detect_faces(crowd_img)

        match_found = False

        best_distance = float("inf")
        best_face = None

        for face in crowd_faces:

            x, y, w, h = face["box"]

            x = abs(x)
            y = abs(y)

            face_crop = crowd_img[y:y+h, x:x+w]

            try:

                embedding = get_embedding(face_crop)

                distance = np.linalg.norm(
                    target_embedding - embedding
                )

                print(distance)

                # Keep only the closest face
                if distance < best_distance:
                    best_distance = distance
                    best_face = (x, y, w, h)

            except:
                pass


        # Draw only one rectangle
        if best_face is not None and best_distance < 0.9:

            x, y, w, h = best_face

            cv2.rectangle(
                crowd_img,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                3
            )

            cv2.putText(
                crowd_img,
                f"Match ({best_distance:.2f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            st.success(f"Match found! Distance = {best_distance:.3f}")

        else:
            st.warning("No matching person found.")


        st.image(
            crowd_img,
            caption="Result Image",
            use_container_width=True
        )

        # Save output
        output_image = cv2.cvtColor(
            crowd_img,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            "output/result.jpg",
            output_image
        )