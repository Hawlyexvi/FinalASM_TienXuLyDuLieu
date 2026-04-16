import argparse
import os

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = ['gia_nha', 'so_phong', 'dien_tich', 'loai_nha', 'mo_ta']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def quick_eda(df: pd.DataFrame) -> None:
    print("--- Thống kê ---")
    print(df.describe(include='all'))
    print("\n--- Missing values ---")
    print(df.isnull().sum())

    if PLOTTING_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df['gia_nha'].dropna(), kde=True, ax=axes[0])
        axes[0].set_title('Phân phối giá nhà')
        sns.boxplot(x=df['dien_tich'].dropna(), ax=axes[1])
        axes[1].set_title('Boxplot diện tích')
        plt.tight_layout()
        plt.show()
    else:
        print("Matplotlib hoặc seaborn không cài đặt; bỏ qua phần vẽ biểu đồ.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    validate_columns(df)

    if not df['so_phong'].mode().empty:
        df['so_phong'] = df['so_phong'].fillna(df['so_phong'].mode()[0])
    else:
        df['so_phong'] = df['so_phong'].fillna(0)

    df['dien_tich'] = df['dien_tich'].fillna(df['dien_tich'].median())
    df = df[(df['gia_nha'] > 0) & (df['so_phong'] > 0)]

    df['loai_nha'] = df['loai_nha'].replace(
        {'Chung cu': 'Chung cư', 'cc': 'Chung cư', 'Nha pho': 'Nhà phố'}
    )
    df = df.drop_duplicates()

    Q1 = df['gia_nha'].quantile(0.25)
    Q3 = df['gia_nha'].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df['gia_nha'] = np.clip(df['gia_nha'], lower, upper)

    scaler = MinMaxScaler()
    df['dien_tich_scaled'] = scaler.fit_transform(df[['dien_tich']])
    df = pd.get_dummies(df, columns=['loai_nha'], prefix='type')

    return df


def vectorize_descriptions(df: pd.DataFrame):
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    return tfidf.fit_transform(df['mo_ta'].fillna(''))


def find_similar_descriptions(tfidf_matrix, threshold: float = 0.9):
    if tfidf_matrix.shape[0] < 2:
        return []
    cosine_sim = cosine_similarity(tfidf_matrix)
    return [
        (i, j, cosine_sim[i, j])
        for i in range(len(cosine_sim))
        for j in range(i + 1, len(cosine_sim))
        if cosine_sim[i, j] > threshold
    ]


def parse_args():
    parser = argparse.ArgumentParser(description='Tiền xử lý dữ liệu bất động sản')
    parser.add_argument(
        '--data-file', '-f',
        default='dataset.csv',
        help='Đường dẫn tới file dữ liệu CSV (mặc định: dataset.csv)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data(args.data_file)
    quick_eda(df)
    df = clean_data(df)
    tfidf_matrix = vectorize_descriptions(df)
    dups = find_similar_descriptions(tfidf_matrix)
    print(f"Phát hiện {len(dups)} cặp mô tả trùng >90%")


if __name__ == '__main__':
    main()
