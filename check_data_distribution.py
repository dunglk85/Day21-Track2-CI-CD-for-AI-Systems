"""
Bonus 5: Canh bao lech lac du lieu (class imbalance check).
Chay TRUOC khi train. Tinh ty le mau cua tung lop (0, 1, 2) trong tap train.
Neu lop nao chiem < 10% tong so mau -> in canh bao ro rang ra log.
Ghi ty le phan phoi ra outputs/class_distribution.json de buoc sau
(generate_report.py) gop vao outputs/metrics.json cung accuracy/f1_score.

Script nay KHONG lam fail pipeline - chi canh bao, vi imbalance khong
nhat thiet la loi (co the do ban chat bai toan). Neu muon chan cung,
doi WARN_ONLY = False.
"""

import json
import os
import sys
import pandas as pd

DATA_PATH = "data/train_phase1.csv"       # Da doi thanh duong dan that
TARGET_COLUMN = "target"           # TODO: doi thanh ten cot nhan that
EXPECTED_CLASSES = [0, 1, 2]
MIN_CLASS_RATIO = 0.10             # nguong 10%
WARN_ONLY = True                   # True: chi canh bao. False: fail pipeline neu imbalance.

OUTPUT_PATH = "outputs/class_distribution.json"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    total = len(df)

    if TARGET_COLUMN not in df.columns:
        print(f"ERROR: Khong tim thay cot '{TARGET_COLUMN}' trong {DATA_PATH}")
        sys.exit(1)

    counts = df[TARGET_COLUMN].value_counts()

    distribution = {}
    any_imbalanced = False

    print("=" * 50)
    print("PHAN PHOI NHAN TRONG TAP TRAIN")
    print("=" * 50)

    for cls in EXPECTED_CLASSES:
        count = int(counts.get(cls, 0))
        ratio = count / total if total > 0 else 0.0
        distribution[str(cls)] = {"count": count, "ratio": round(ratio, 4)}

        flag = ""
        if ratio < MIN_CLASS_RATIO:
            flag = "  <-- CANH BAO: duoi nguong 10%"
            any_imbalanced = True

        print(f"  Lop {cls}: {count:>6} mau  ({ratio * 100:.2f}%){flag}")

    print("=" * 50)

    if any_imbalanced:
        print("WARNING: Phat hien mat can bang du lieu (data imbalance)!")
        print("Mot hoac nhieu lop chiem duoi 10% tong so mau.")
        print("Can nhac: oversampling, undersampling, class_weight, hoac thu thap them du lieu.")
    else:
        print("OK: Phan phoi nhan tuong doi can bang giua cac lop.")

    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "total_samples": total,
                "class_distribution": distribution,
                "is_imbalanced": any_imbalanced,
                "imbalance_threshold": MIN_CLASS_RATIO,
            },
            f,
            indent=2,
        )
    print(f"\nDa ghi phan phoi nhan vao {OUTPUT_PATH}")

    if any_imbalanced and not WARN_ONLY:
        print("FAILED: Pipeline bi chan do du lieu mat can bang (WARN_ONLY=False).")
        sys.exit(1)


if __name__ == "__main__":
    main()
