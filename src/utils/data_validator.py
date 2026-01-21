import pandas as pd

def validate_numeric_columns(df, context=""):
    """
    Check for non-numeric columns that could cause conversion errors.

    Args:
        df: pandas DataFrame to validate
        context: string describing the context for error message

    Raises:
        ValueError: if non-numeric columns are found
    """
    non_numeric_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric to see if it fails
            try:
                pd.to_numeric(df[col])
            except (ValueError, TypeError):
                non_numeric_cols.append(col)

    if non_numeric_cols:
        context_msg = f" in {context}" if context else ""
        raise ValueError(
            f"The following columns contain non-numeric values that cannot be converted to float{context_msg}: {non_numeric_cols}\n"
            f"Please preprocess your data using appropriate transformers such as:\n"
            f"- transformers.one_hot_encoder for categorical variables\n"
            f"- transformers.ordinal_encoder for ordinal categorical variables\n"
            f"- transformers.label_encode for target variables\n"
            f"Add these transformers to your configuration file under the [preprocess] section."
        )
