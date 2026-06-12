import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras_facenet import FaceNet

# ---------------- Load FaceNet ----------------
embedder = FaceNet()

# Optional: load custom weights
embedder.model.load_weights("models/facenet.weights.h5")

# OpenCV Face Detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ---------------- Streamlit UI ----------------
st.set_page_config(
    page_title="Face Detection in Crowd",
    page_icon="🧑",
    layout="wide"
)

st.title("🧑 Face Detection in Crowd Image")
st.write(
    "Upload a target person's image and a crowd image. "
    "The matching person will be highlighted."
)

# Upload images
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

    # ========= Target Image =========
    target_img = np.array(Image.open(target_file))

    gray_target = cv2.cvtColor(
        target_img,
        cv2.COLOR_RGB2GRAY
    )

    target_faces = face_detector.detectMultiScale(
        gray_target,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(target_faces) == 0:

        st.error("No face detected in target image!")

    else:

        x, y, w, h = target_faces[0]

        target_face = target_img[y:y+h, x:x+w]

        target_embedding = get_embedding(target_face)

        # ========= Crowd Image =========
        crowd_img = np.array(Image.open(crowd_file))

        gray_crowd = cv2.cvtColor(
            crowd_img,
            cv2.COLOR_RGB2GRAY
        )

        crowd_faces = face_detector.detectMultiScale(
            gray_crowd,
            scaleFactor=1.1,
            minNeighbors=5
        )

        best_similarity = -1
        best_face = None

        # Find most similar face
        for (x, y, w, h) in crowd_faces:

            face_crop = crowd_img[y:y+h, x:x+w]

            try:

                embedding = get_embedding(face_crop)

                # Cosine similarity
                similarity = np.dot(
                    target_embedding,
                    embedding
                ) / (
                    np.linalg.norm(target_embedding)
                    * np.linalg.norm(embedding)
                )

                print("Similarity:", similarity)

                if similarity > best_similarity:

                    best_similarity = similarity
                    best_face = (x, y, w, h)

            except:
                pass

        # Draw only one rectangle
        if best_face is not None and best_similarity > 0.75:

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
                f"Match ({best_similarity:.2f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            st.success(
                f"Match Found! Similarity = {best_similarity:.3f}"
            )

        else:

            st.warning("No matching person found.")

        # Show output image
        st.image(
            crowd_img,
            caption="Detected Person",
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