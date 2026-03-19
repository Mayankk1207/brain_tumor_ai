import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import random
import cv2

# ==============================
# PATHS
# ==============================
RAW_DIR = "/home/mayank/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

OUT_DIR = "data_processed"

TRAIN_IMG = os.path.join(OUT_DIR, "train/images")
TRAIN_MASK = os.path.join(OUT_DIR, "train/masks")
VAL_IMG = os.path.join(OUT_DIR, "val/images")
VAL_MASK = os.path.join(OUT_DIR, "val/masks")

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_MASK, exist_ok=True)
os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_MASK, exist_ok=True)

# ==============================
# SETTINGS
# ==============================
MAX_PATIENTS = 120
TRAIN_SPLIT = 0.8
IMG_SIZE = 256

# ==============================
# NORMALIZATION
# ==============================
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

# ==============================
# FIND PATIENTS
# ==============================
patients = []

for root, _, files in os.walk(RAW_DIR):
    flair = t1 = t1ce = t2 = seg = None

    for f in files:
        path = os.path.join(root, f)

        if "_flair" in f:
            flair = path
        elif "_t1ce" in f:
            t1ce = path
        elif "_t1.nii" in f:
            t1 = path
        elif "_t2" in f:
            t2 = path
        elif "_seg" in f:
            seg = path

    if flair and t1 and t1ce and t2 and seg:
        patients.append((flair, t1, t1ce, t2, seg))

print(f"Found {len(patients)} patients")

# ==============================
# SPLIT
# ==============================
random.shuffle(patients)
patients = patients[:MAX_PATIENTS]

split = int(len(patients) * TRAIN_SPLIT)
train_patients = patients[:split]
val_patients = patients[split:]

# ==============================
# PROCESS FUNCTION
# ==============================
def process(patients, img_dir, mask_dir):
    count = 0

    for flair_p, t1_p, t1ce_p, t2_p, seg_p in tqdm(patients):
        try:
            flair = normalize(nib.load(flair_p).get_fdata())
            t1    = normalize(nib.load(t1_p).get_fdata())
            t1ce  = normalize(nib.load(t1ce_p).get_fdata())
            t2    = normalize(nib.load(t2_p).get_fdata())
            seg   = nib.load(seg_p).get_fdata()

            for i in range(flair.shape[2]):
                f  = flair[:, :, i]
                t1s = t1[:, :, i]
                t1ces = t1ce[:, :, i]
                t2s = t2[:, :, i]
                m  = seg[:, :, i]

                # FIX LABELS (IMPORTANT)
                m[m == 4] = 3

                # Skip most empty slices but keep some
                if np.max(m) == 0:
                    if random.random() > 0.1:
                        continue

                # Resize
                f = cv2.resize(f, (IMG_SIZE, IMG_SIZE))
                t1s = cv2.resize(t1s, (IMG_SIZE, IMG_SIZE))
                t1ces = cv2.resize(t1ces, (IMG_SIZE, IMG_SIZE))
                t2s = cv2.resize(t2s, (IMG_SIZE, IMG_SIZE))
                m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

                # Stack channels (4,H,W)
                img = np.stack([f, t1s, t1ces, t2s], axis=0)

                np.save(os.path.join(img_dir, f"{count}.npy"), img.astype(np.float32))
                np.save(os.path.join(mask_dir, f"{count}.npy"), m.astype(np.uint8))

                count += 1

        except Exception as e:
            print("Error:", e)

    return count


# ==============================
# RUN
# ==============================
print("Processing TRAIN...")
train_count = process(train_patients, TRAIN_IMG, TRAIN_MASK)

print("Processing VAL...")
val_count = process(val_patients, VAL_IMG, VAL_MASK)

print("DONE")
print("Train:", train_count)
print("Val:", val_count)