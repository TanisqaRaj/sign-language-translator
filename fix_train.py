import re
path = "train_model.py"
with open(path, "rb") as f:
    data = f.read()
print("before:", data.count(b"HYBRID_FEATURES"))
data = re.sub(rb"\r?\n    HYBRID_FEATURES,", b"", data)
data = data.replace(b"input_dim: int = HYBRID_FEATURES", b"input_dim: int = LANDMARK_FEATURES")
data = data.replace(b"HYBRID_FEATURES", b"LANDMARK_FEATURES")
data = data.replace(
    b"Re-run preprocess.py with the hybrid pipeline to regenerate landmarks.csv.",
    b"Re-run preprocess.py to regenerate landmarks.csv.",
)
print("after:", data.count(b"HYBRID_FEATURES"))
with open(path, "wb") as f:
    f.write(data)
print("saved")
