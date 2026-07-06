"""
Bonus 4: Hoan tra ve phien ban truoc (rollback safety gate).
Truoc khi deploy: tai outputs/metrics.json cua LAN DEPLOY THANH CONG GAN NHAT
tu Cloud Storage (production/metrics.json). So sanh accuracy moi vs accuracy cu.
Chi cho phep deploy khi accuracy moi >= accuracy cu.
Ghi lai ket qua so sanh ro rang vao log pipeline.

Exit code:
  0 = duoc phep deploy
  1 = KHONG duoc phep deploy (accuracy giam, hoac loi khong the xac nhan an toan)
"""

import json
import sys
import os
from google.cloud import storage
from google.api_core.exceptions import NotFound

BUCKET_NAME = os.environ["CLOUD_BUCKET"]
PROD_METRICS_BLOB = "production/metrics.json"
CURRENT_METRICS_PATH = "outputs/metrics.json"


def main() -> None:
    with open(CURRENT_METRICS_PATH) as f:
        current_metrics = json.load(f)
    current_acc = float(current_metrics["accuracy"])

    print("=" * 50)
    print("SO SANH VOI PHIEN BAN DANG CHAY PRODUCTION")
    print("=" * 50)
    print(f"Accuracy model MOI: {current_acc:.4f}")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(PROD_METRICS_BLOB)

    try:
        prev_metrics_raw = blob.download_as_text()
        prev_metrics = json.loads(prev_metrics_raw)
        prev_acc = float(prev_metrics["accuracy"])
        print(f"Accuracy model dang chay PRODUCTION: {prev_acc:.4f}")
    except NotFound:
        print("Chua co model nao dang chay production truoc do (lan deploy dau tien).")
        print("=> Cho phep deploy.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Khong the tai metrics.json tu Cloud Storage: {e}")
        print("=> Khong the xac nhan an toan, CHAN deploy de tranh rui ro.")
        sys.exit(1)

    print("-" * 50)
    if current_acc >= prev_acc:
        diff = current_acc - prev_acc
        print(f"KET QUA: PASSED. Model moi ({current_acc:.4f}) >= production ({prev_acc:.4f}). "
              f"Chenh lech: +{diff:.4f}")
        print("=> Cho phep deploy.")
        sys.exit(0)
    else:
        diff = prev_acc - current_acc
        print(f"KET QUA: FAILED. Model moi ({current_acc:.4f}) < production ({prev_acc:.4f}). "
              f"Giam: -{diff:.4f}")
        print("=> HUY DEPLOY. Giu nguyen model production hien tai.")
        sys.exit(1)


if __name__ == "__main__":
    main()
