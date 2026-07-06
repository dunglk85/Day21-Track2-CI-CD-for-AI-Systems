"""
Data validation gate su dung Great Expectations.
Chay TRUOC buoc train de chan som du lieu loi (null bat thuong, sai schema,
gia tri ngoai khoang, so dong qua it, v.v.)

Fail (exit 1) neu bat ky expectation nao khong pass -> pipeline dung lai,
khong lang phi compute cho buoc train.
"""

import sys
import pandas as pd
import great_expectations as gx


DATA_PATH = "data/train.csv"  # TODO: doi thanh duong dan du lieu that cua ban


def build_suite() -> "gx.ExpectationSuite":
    """Dinh nghia cac rule kiem tra du lieu. Tuy chinh theo dataset that."""
    suite = gx.ExpectationSuite(name="training_data_suite")

    # --- Schema / ton tai cot ---
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=["feature_1", "feature_2", "feature_3", "target"],
            exact_match=False,  # cho phep co them cot khac ngoai danh sach nay
        )
    )

    # --- Khong duoc null o cot quan trong ---
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="target")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="feature_1", mostly=0.95  # cho phep toi da 5% null
        )
    )

    # --- Gia tri trong khoang hop ly ---
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="feature_2", min_value=0, max_value=1
        )
    )

    # --- Kieu du lieu cua nhan (vi du bai toan phan loai nhi phan) ---
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="target", value_set=[0, 1])
    )

    # --- Khong duoc trung lap toan bo dong ---
    suite.add_expectation(
        gx.expectations.ExpectCompoundColumnsToBeUnique(
            column_list=["feature_1", "feature_2", "feature_3"]
        )
    )

    # --- So luong dong toi thieu (phat hien pull data bi thieu/loi) ---
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=100)
    )

    return suite


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    context = gx.get_context()

    data_source = context.data_sources.add_pandas("runtime_data_source")
    data_asset = data_source.add_dataframe_asset(name="training_data")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = build_suite()
    results = batch.validate(suite)

    # In ket qua chi tiet
    passed = 0
    failed = 0
    for r in results.results:
        status = "PASS" if r.success else "FAIL"
        expectation_type = r.expectation_config.type
        print(f"[{status}] {expectation_type}")
        if not r.success:
            failed += 1
            print(f"       details: {r.result}")
        else:
            passed += 1

    print(f"\nSummary: {passed} passed, {failed} failed")

    if not results.success:
        print("\nData validation FAILED. Blocking pipeline before training.")
        sys.exit(1)

    print("\nData validation PASSED.")


if __name__ == "__main__":
    main()
