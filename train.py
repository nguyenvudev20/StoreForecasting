import argparse
import pandas as pd

from utils import (
    validate_columns,
    preprocess_training_data,
    train_valid_split_by_date,
    fit_model,
    evaluate_model,
    save_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="train.csv", help="Đường dẫn file CSV")
    parser.add_argument("--valid_days", type=int, default=90, help="Số ngày validation cuối")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    df_feat = preprocess_training_data(df)
    train_df, valid_df, split_date = train_valid_split_by_date(df_feat, valid_days=args.valid_days)

    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("Không đủ dữ liệu để chia train/validation.")

    model = fit_model(train_df)
    metrics, _ = evaluate_model(model, valid_df)

    print("===== TRAIN SUCCESS =====")
    print(f"Validation from : {split_date.date()}")
    print(f"MAE             : {metrics['MAE']:.4f}")
    print(f"RMSE            : {metrics['RMSE']:.4f}")
    print(f"R2              : {metrics['R2']:.4f}")
    print(f"MAPE            : {metrics['MAPE']:.2f}%")
    print(f"SMAPE           : {metrics['SMAPE']:.2f}%")

    save_model(model)
    print("Model saved to model/model.pkl")
    print("Features saved to model/features.pkl")


if __name__ == "__main__":
    main()