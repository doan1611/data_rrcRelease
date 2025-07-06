import os
import numpy as np
from data_util import load_data, convert_to_csv

if __name__ == "__main__":
    raw_dir = "data/raw"
    window_size = 10
    num_features = 4
    stride = 1
    out_dir = "data/dataprocess"

    os.makedirs(out_dir, exist_ok=True)
    print("Loading data...")
    #sessions = load_and_filter_sessions(raw_dir)

    print("Normalizing and windowing data...")
    #X, y = normalize_and_window_sessions(sessions, window_size, stride)
    X, y, scaler = load_data(raw_dir, window_size, stride)
    print(f"Done! Total samples: {len(X)}")

    #print(scaler)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]

    # Reshape data
    X = X.reshape((X.shape[0], window_size * num_features))
    y = y.reshape((-1, 1))

    print(X.shape)
    np.save(os.path.join(out_dir, "X_all.npy"), X)
    np.save(os.path.join(out_dir, "y_all.npy"), y)

    convert_to_csv(out_dir, X, "X_all.csv")
    convert_to_csv(out_dir, y, "y_all.csv")

    print(f"Done! Total samples after removal: {len(X)}")
    print("X has NaNs:", np.isnan(X).sum())
    print("y has NaNs:", np.isnan(y).sum())
