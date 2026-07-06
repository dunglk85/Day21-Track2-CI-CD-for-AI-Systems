import pytest
import pandas as pd
import numpy as np
import great_expectations as gx
from validate_data import build_suite


COLUMN_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type", "target",
]


def _make_clean_df(n=200):
    rng = np.random.default_rng(42)
    data = {col: rng.random(n) for col in COLUMN_NAMES[:12]}
    data["wine_type"] = rng.integers(0, 2, size=n)
    data["target"] = rng.integers(0, 3, size=n)
    return pd.DataFrame(data)


def _validate_df(df):
    context = gx.get_context()
    data_source = context.data_sources.add_pandas("test_data_source")
    data_asset = data_source.add_dataframe_asset(name="test_data")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    suite = build_suite()
    return batch.validate(suite)


@pytest.fixture(autouse=True)
def gx_context():
    context = gx.get_context()
    return context


class TestBuildSuite:
    def test_returns_suite_with_correct_name(self):
        suite = build_suite()
        assert suite.name == "training_data_suite"

    def test_suite_has_expected_expectations(self):
        suite = build_suite()
        names = [e.expectation_type for e in suite.expectations]
        assert "expect_table_columns_to_match_set" in names
        assert "expect_column_values_to_not_be_null" in names
        assert "expect_column_values_to_be_between" in names
        assert "expect_column_values_to_be_in_set" in names
        assert "expect_table_row_count_to_be_between" in names

    def test_column_set_matches(self):
        suite = build_suite()
        for e in suite.expectations:
            if e.expectation_type == "expect_table_columns_to_match_set":
                assert set(e.column_set) == set(COLUMN_NAMES)
                assert e.exact_match is True


class TestValidationOnCleanData:
    def test_all_expectations_pass(self):
        df = _make_clean_df()
        results = _validate_df(df)
        assert results.success is True

    def test_validates_correct_row_count(self):
        df = _make_clean_df(n=200)
        results = _validate_df(df)
        for r in results.results:
            if r.expectation_config.type == "expect_table_row_count_to_be_between":
                assert r.success is True


class TestValidationOnBadData:
    def test_fails_on_null_target(self):
        df = _make_clean_df()
        df.loc[0, "target"] = None
        results = _validate_df(df)
        assert results.success is False

    def test_fails_on_missing_column(self):
        df = _make_clean_df()
        df = df.drop(columns=["fixed_acidity"])
        results = _validate_df(df)
        assert results.success is False
        column_failures = [
            r for r in results.results
            if r.expectation_config.type == "expect_table_columns_to_match_set"
        ]
        assert any(not r.success for r in column_failures)

    def test_fails_on_extra_column(self):
        df = _make_clean_df()
        df["extra_col"] = 0
        results = _validate_df(df)
        assert results.success is False

    def test_fails_on_pH_out_of_range(self):
        df = _make_clean_df()
        df.loc[0, "pH"] = 15.0
        results = _validate_df(df)
        assert results.success is False

    def test_fails_on_target_out_of_set(self):
        df = _make_clean_df()
        df.loc[0, "target"] = 99
        results = _validate_df(df)
        assert results.success is False

    def test_fails_on_too_few_rows(self):
        df = _make_clean_df(n=50)
        results = _validate_df(df)
        assert results.success is False
