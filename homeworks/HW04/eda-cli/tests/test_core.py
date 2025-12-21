from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    """Базовый тестовый датафрейм без проблемных колонок."""
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0
    
    # Проверяем, что для обычного датафрейма флаги False
    assert flags['has_constant_columns'] == False
    assert flags['has_high_cardinality_categoricals'] == False


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_compute_quality_flags_has_constant_columns_true():
    """Тест: обнаружение константных колонок."""
    # Создаем отдельный датафрейм для этого теста
    df = pd.DataFrame({
        "age": [10, 20, 30, None],
        "height": [140, 150, 160, 170],
        "city": ["A", "B", "A", None],
        'constant_col': [42, 42, 42, 42],  # 4 значения, как в других колонках
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags['has_constant_columns'] == True
    
    constant_col_summary = next(c for c in summary.columns if c.name == "constant_col")
    assert constant_col_summary.unique == 1


def test_compute_quality_flags_has_high_cardinality_categoricals_true():
    """Тест: обнаружение категориальных колонок с высокой кардинальностью."""
    # Нужно больше строк, чтобы было >10 уникальных значений
    df = pd.DataFrame({
        "age": list(range(15)),  # 15 строк
        "city": [f"city_{i}" for i in range(15)],  # 15 уникальных значений > 10
        "normal_col": [1] * 15,  # Константная колонка
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Оба флага должны быть True
    assert flags['has_constant_columns'] == True
    assert flags['has_high_cardinality_categoricals'] == True
    
    high_card_col = next(c for c in summary.columns if c.name == "city")
    assert high_card_col.unique > 10
    assert not high_card_col.is_numeric


def test_compute_quality_flags_numeric_not_high_cardinality():
    """Тест: числовые колонки с высокой уникальностью не считаются."""
    df = pd.DataFrame({
        "numeric_high_unique": list(range(15)),  # 15 уникальных числовых значений
        "category_low": ["A", "B"] * 7 + ["A"],  # только 2 уникальных
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Должно быть False, т.к. высокая уникальность только у числовой колонки
    assert flags['has_high_cardinality_categoricals'] == False