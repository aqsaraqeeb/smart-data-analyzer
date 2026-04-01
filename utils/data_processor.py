"""
data_processor.py
-----------------
Handles all data loading, cleaning, and statistical analysis.
This module is the backbone of the Smart Data Analyzer.
"""

import pandas as pd
import numpy as np


def load_csv(uploaded_file):
    """
    Load a CSV file uploaded via Streamlit's file uploader.
    Returns a DataFrame or raises an error with a helpful message.
    """
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            raise ValueError("The uploaded CSV file is empty. Please upload a file with data.")
        return df
    except pd.errors.ParserError:
        raise ValueError("Could not parse the file. Please ensure it is a valid CSV.")
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")


def get_basic_info(df):
    """
    Returns a dictionary of basic dataset information:
    - shape, column names, dtypes, missing values
    """
    info = {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return info


def get_statistics(df):
    """
    Returns descriptive statistics for all numeric columns:
    mean, median, std, min, max, quartiles.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None

    stats = numeric_df.describe().T  # Transpose for better readability
    stats["median"] = numeric_df.median()
    stats["skewness"] = numeric_df.skew().round(3)
    stats["kurtosis"] = numeric_df.kurtosis().round(3)
    return stats.round(3)


def get_numeric_columns(df):
    """Returns list of numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df):
    """Returns list of categorical/object column names."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_column_value_counts(df, column, top_n=10):
    """Returns top N value counts for a categorical column."""
    return df[column].value_counts().head(top_n)


def build_summary_text(df):
    """
    Builds a concise text summary of the dataset.
    This is fed into the AI model for generating insights.
    """
    info = get_basic_info(df)
    stats = get_statistics(df)
    cat_cols = get_categorical_columns(df)
    num_cols = get_numeric_columns(df)

    summary_parts = [
        f"Dataset has {info['num_rows']} rows and {info['num_cols']} columns.",
        f"Numeric columns ({len(num_cols)}): {', '.join(num_cols) if num_cols else 'None'}.",
        f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols) if cat_cols else 'None'}.",
        f"Total missing values: {sum(info['missing_values'].values())}.",
        f"Duplicate rows: {info['duplicate_rows']}.",
    ]

    if stats is not None:
        for col in stats.index[:5]:  # Limit to first 5 numeric cols
            row = stats.loc[col]
            summary_parts.append(
                f"Column '{col}': mean={row.get('mean', 'N/A')}, "
                f"std={row.get('std', 'N/A')}, "
                f"min={row.get('min', 'N/A')}, "
                f"max={row.get('max', 'N/A')}."
            )

    if cat_cols:
        for col in cat_cols[:3]:  # Limit to first 3 categorical cols
            top_vals = df[col].value_counts().head(3)
            summary_parts.append(
                f"Column '{col}' top values: {top_vals.to_dict()}."
            )

    return " ".join(summary_parts)


def detect_important_columns(df):
    """
    Auto-detects potentially important columns based on heuristics:
    - High variance numeric columns
    - Low cardinality categorical columns (good for grouping)
    """
    important = {"numeric": [], "categorical": []}

    num_cols = get_numeric_columns(df)
    if num_cols:
        variances = df[num_cols].var().sort_values(ascending=False)
        important["numeric"] = variances.head(3).index.tolist()

    cat_cols = get_categorical_columns(df)
    for col in cat_cols:
        cardinality = df[col].nunique()
        if 2 <= cardinality <= 20:  # Good for grouping/viz
            important["categorical"].append(col)

    return important
