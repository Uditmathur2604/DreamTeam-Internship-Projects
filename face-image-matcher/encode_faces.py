import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import insightface

# Initialize model
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)  # CPU = -1, GPU = 0

# Paths
dataset_dir = "event_photos"
embedding_list = []
filename_list = []

# Process all images
for filename in tqdm(os.listdir(dataset_dir)):
    filepath = os.path.join(dataset_dir, filename)
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img = cv2.imread(filepath)
    if img is None:
        print(f"⚠️ Failed to read {filepath}")
        continue

    faces = model.get(img)
    if not faces:
        print(f"❌ No face in: {filename}")
        continue

    for face in faces:
        embedding_list.append(face.embedding)
        filename_list.append(filename)  # One entry per face

# Save embeddings
embeddings = np.array(embedding_list).astype("float32")
np.save("embeddings.npy", embeddings)

# Save filenames
with open("filenames.pkl", "wb") as f:
    pickle.dump(filename_list, f)

print(f"\n✅ Encoded {len(embedding_list)} faces from {len(set(filename_list))} images.")
