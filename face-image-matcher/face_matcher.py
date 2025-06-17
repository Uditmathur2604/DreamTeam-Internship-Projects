import cv2
import insightface
import numpy as np
import faiss
import pickle
import os
import shutil
from sklearn.preprocessing import normalize

# Load InsightFace model
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)  # CPU=-1, GPU=0

# Load dataset
embeddings = np.load("embeddings.npy")
with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Normalize for cosine similarity
embeddings = normalize(embeddings, axis=1)

# Build FAISS cosine similarity index
index = faiss.IndexFlatIP(512)
index.add(embeddings)

# Webcam capture
cap = cv2.VideoCapture(0)
print("📷 Press 's' to capture image for search...")

frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Webcam - Press 's' to capture", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()

# Extract embedding from webcam image
faces = model.get(frame)
if not faces:
    print("❌ No face detected in webcam image.")
    exit()

query_embedding = normalize(faces[0].embedding.reshape(1, -1), axis=1)

# Search all
D, I = index.search(query_embedding, len(filenames))
threshold = 0.3  # cosine similarity threshold (higher = more similar)

# Create folder for matched results
matched_dir = "matched_faces"
os.makedirs(matched_dir, exist_ok=True)

match_count = 0
for score, idx in zip(D[0], I[0]):
    if score < threshold:
        continue  # ignore poor matches

    img_file = filenames[idx].strip()
    img_path = os.path.join("event_photos", img_file)

    if not os.path.isfile(img_path):
        print(f"❌ File not found: {img_path}")
        continue

    # Copy matched image
    shutil.copy(img_path, os.path.join(matched_dir, img_file))

    # Show image (resized for display, not zoomed)
    img = cv2.imread(img_path)
    if img is not None:
        resized = cv2.resize(img, (600, 400))
        cv2.imshow(f"Match {match_count + 1}", resized)

    print(f"✅ Match {match_count + 1}: {img_file} - Similarity: {score:.4f}")
    match_count += 1

if match_count == 0:
    print("❌ No matches found above threshold.")
else:
    print(f"\n📁 {match_count} images saved to '{matched_dir}'")

cv2.waitKey(0)
cv2.destroyAllWindows()



# import cv2
# import insightface
# import numpy as np
# import faiss
# import pickle
# import os
# from pathlib import Path
# from datetime import datetime

# # Paths
# DATASET_DIR = "event_photos"
# EMBEDDING_PATH = "embeddings.npy"
# FILENAME_PATH = "filenames.pkl"
# USERS_DIR = "retrieved_users"

# # Threshold for face match (tune based on accuracy)
# DISTANCE_THRESHOLD = 300  # Lower = stricter matching

# # Load InsightFace model
# model = insightface.app.FaceAnalysis(name='buffalo_l')
# model.prepare(ctx_id=0)  # Use 0 for CPU, 0 if CUDA GPU available

# # Load embeddings and filenames
# embeddings = np.load(EMBEDDING_PATH)
# with open(FILENAME_PATH, "rb") as f:
#     filenames = pickle.load(f)

# # Build FAISS index
# index = faiss.IndexFlatL2(512)
# index.add(embeddings)

# # Webcam input
# cap = cv2.VideoCapture(0)
# print("📷 Press 's' to capture image for search...")

# frame = None
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue
#     cv2.imshow("Webcam - Press 's' to capture", frame)
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # Detect face from webcam
# faces = model.get(frame)
# if not faces:
#     print("❌ No face detected in webcam image.")
#     exit()

# query_embedding = faces[0].embedding.reshape(1, -1)

# # Perform similarity search (return top 100)
# k = 100
# D, I = index.search(query_embedding, k)

# # Filter matches based on threshold
# matched_filenames = []
# for dist, idx in zip(D[0], I[0]):
#     if dist <= DISTANCE_THRESHOLD:
#         matched_filenames.append(filenames[idx].strip())

# if not matched_filenames:
#     print("❌ No matching faces found in dataset.")
#     exit()

# # Hash face to generate consistent user ID
# face_hash = str(np.round(query_embedding[0], 2).tolist())
# user_id = f"user_{abs(hash(face_hash)) % 1000000}"
# user_folder = Path(USERS_DIR) / user_id
# user_folder.mkdir(parents=True, exist_ok=True)

# print(f"✅ Found {len(matched_filenames)} matches for {user_id}")
# print(f"📁 Storing results in: {user_folder}")

# # Copy matched images (avoid duplicates)
# for filename in matched_filenames:
#     src = os.path.join(DATASET_DIR, filename)
#     dst = user_folder / filename

#     if not os.path.isfile(src):
#         print(f"❌ File missing: {src}")
#         continue

#     if not dst.exists():
#         img = cv2.imread(src)
#         cv2.imwrite(str(dst), img)
#         print(f"📄 Saved: {dst.name}")
#     else:
#         print(f"📂 Already exists: {dst.name}")

# # Display all retrieved images
# for filename in matched_filenames:
#     img_path = user_folder / filename
#     img = cv2.imread(str(img_path))
#     if img is not None:
#         resized = cv2.resize(img, (600, 400))
#         cv2.imshow(f"Match - {filename}", resized)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
