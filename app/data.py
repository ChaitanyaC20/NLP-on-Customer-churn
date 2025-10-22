import pandas as pd
from app.config import DATA_PATH

def load_sentiment_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.replace(" ", "_")
    assert "Generated_Reviews" in df.columns, "Missing 'Generated_Reviews'"
    assert "Satisfaction_Score" in df.columns, "Missing 'Satisfaction_Score'"
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    return df