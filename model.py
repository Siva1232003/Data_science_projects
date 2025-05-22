import h5py

model_path = "fake_product_detection_model.h5"

with h5py.File(model_path, "r") as f:
    keras_version = f.attrs.get("keras_version", "Unknown")
    backend = f.attrs.get("backend", "Unknown")
    
    print(f"Keras version: {keras_version}")
    print(f"Backend: {backend}")
