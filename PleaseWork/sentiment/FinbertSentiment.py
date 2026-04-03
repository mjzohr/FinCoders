from transformers import pipeline
from SentimentAnalysisBase import SentimentAnalysisBase
import pandas as pd  # Ensure pandas is imported if not already


class FinbertSentiment(SentimentAnalysisBase):

    def __init__(self):
        # Initialize the FinBERT sentiment analysis pipeline
        self._sentiment_analysis = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert")
        super().__init__()

    def calc_sentiment_score(self, df):
        # Ensure df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Check for the 'title' column
        if 'title' not in df.columns:
            raise ValueError("DataFrame must contain a 'title' column.")

        # Apply sentiment analysis
        df['sentiment'] = df['title'].apply(self._sentiment_analysis)

        # Extract sentiment score
        df['sentiment_score'] = df['sentiment'].apply(
            lambda x: (-1 if x[0]['label'] == 'negative' else
                       1 if x[0]['label'] == 'positive' else 0) * x[0]['score']
        )

        self.df = df  # Save to the object's DataFrame
