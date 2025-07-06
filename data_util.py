import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1️⃣ — Chuyển thời gian sang mili giây
def convert_time_to_millis(df, time_col="Time"):
    def parse_time_str(s):
        try:
            mins, rest = s.split(":")
            secs = float(rest)
            return int(float(mins) * 60 * 1000 + secs * 1000)
        except:
            return np.nan

    df[time_col] = df[time_col].astype(str).str.strip()
    df[time_col] = df[time_col].apply(parse_time_str)
    df = df.dropna(subset=[time_col])
    df[time_col] = df[time_col].astype(int)
    return df

# 2️⃣ — Gộp tất cả file và thêm dòng "end"
def load_and_merge_all_files(raw_dir, feature_cols, time_col="Time"):
    dfs = []
    for fname in os.listdir(raw_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(raw_dir, fname))
            df = convert_time_to_millis(df, time_col)
            dfs.append(df)

            # Thêm dòng "end" để phân cách
            end_row = {col: 0 for col in df.columns}
            end_row["UlPktCnt"] = "end"
            dfs.append(pd.DataFrame([end_row]))

    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

# 3️⃣ — Chèn dòng 0 tại rrcRelease (không áp dụng cho "end")
def insert_zero_rows_rrc(df, time_col="Time", interval_ms=500):
    print("🔄 Đang chèn dòng 0 tại RRC Release...")
    feature_cols = ["UlPktCnt", "UlPktBytes", "DlPktCnt", "DlPktBytes"]
    rows = []
    fake_times = set()
    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]
        if "rrcReleas" in str(row["UlPktCnt"]):
            if 0 < i < len(df) - 1:
                t1 = df.iloc[i - 1][time_col]
                t2 = df.iloc[i + 1][time_col]
                n_fill = int((t2 - t1) // interval_ms) - 1
                for j in range(n_fill):
                    fake_time = t1 + (j + 1) * interval_ms
                    fake_row = {col: 0 for col in feature_cols}
                    fake_row[time_col] = fake_time
                    fake_times.add(fake_time)
                    rows.append(fake_row)
            i += 1
        else:
            rows.append(row.to_dict())
            i += 1

    df_filled = pd.DataFrame(rows)
    return df_filled, fake_times


# 4️⃣ — Chuẩn hóa dữ liệu và xử lý dòng "end"
def normalize_features(df, feature_cols):
    is_end_row = df["UlPktCnt"].astype(str).str.lower() == "end"
    df.loc[is_end_row, feature_cols] = 0
    #df.to_csv("output.csv", index=False)
    print(df[feature_cols].dtypes)
    df[feature_cols] = df[feature_cols].astype(float)
    print(df[feature_cols].dtypes)
    # 🎯 Lọc dòng hợp lệ để fit scaler
    valid_mask = (df[feature_cols] >= 0) & (df[feature_cols] <= 1e9)
    valid_rows = valid_mask.all(axis=1)

    scaler = MinMaxScaler()
    scaler.fit(df.loc[valid_rows, feature_cols])  # chỉ fit trên dòng hợp lệ

    # Áp dụng transform cho toàn bộ (kể cả dòng nhiễu)
    df[feature_cols] = scaler.transform(df[feature_cols])
    df["is_end"] = is_end_row
    return df, scaler

# 5️⃣ — Tạo sliding window và bỏ window kết thúc bằng dòng giả hoặc "end"
def create_windows(df, feature_cols, label_col, fake_times, time_col="Time", window_size=40, stride=1):
    print("🔄 Đang tạo sliding windows...")
    features = df[feature_cols].values
    labels = df[label_col].values
    times = df[time_col].values
    is_end_flags = df["is_end"].values

    X, y = [], []
    for i in range(0, len(df) - window_size + 1, stride):
        end_time = times[i + window_size - 1]
        window = features[i:i + window_size]
        if end_time in fake_times:
            continue
        if np.any(is_end_flags[i:i + window_size]):
            continue
        if np.any(window < 0) or np.any(window > 1):
            continue
        X.append(features[i:i + window_size])
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

# 6️⃣ — Hàm tổng hợp
def load_data(raw_dir, window_size=40, stride=1):
    feature_cols = ["UlPktCnt", "UlPktBytes", "DlPktCnt", "DlPktBytes"]
    label_col = "Expected prediction"
    time_col = "Time"

    df = load_and_merge_all_files(raw_dir, feature_cols, time_col)
    df, fake_times = insert_zero_rows_rrc(df, time_col)
    df, scaler = normalize_features(df, feature_cols)
    X, y = create_windows(df, feature_cols, label_col, fake_times, time_col, window_size, stride)
    return X, y, scaler

def convert_to_csv(out_directory, file, save_name):
    print("    - convert to csv")
    df = pd.DataFrame(file)
    df.to_csv(os.path.join(out_directory, save_name), index=False)
    print("Data has been saved to ", save_name)
