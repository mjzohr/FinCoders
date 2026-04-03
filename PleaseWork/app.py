import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import sentiment
from flask import Flask, render_template, request
from plotly.utils import PlotlyJSONEncoder
from sentiment.FinbertSentiment import FinbertSentiment
from yahoo_api import API  # Assuming this is a custom module you created or have access to

EST = pytz.timezone('US/Eastern')

app = Flask(__name__)

# Ensure the FinbertSentiment class works properly
sentimentAlgo = sentiment()


# Fetch price history for the stock ticker after a certain date
def get_price_history(ticker: str, earliest_datetime: pd.Timestamp) -> pd.DataFrame:
    try:
        return API.get_price_history(ticker, earliest_datetime)
    except Exception as e:
        print(f"Error fetching price history for {ticker}: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on error


# Fetch news data for the stock ticker
def get_news(ticker: str) -> pd.DataFrame:
    try:
        sentimentAlgo.set_symbol(ticker)
        return API.get_news(ticker)
    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on error


# Calculate sentiment scores for the news articles
def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    try:
        sentimentAlgo.set_data(news_df)
        sentimentAlgo.calc_sentiment_score()
        return sentimentAlgo.df
    except Exception as e:
        print(f"Error calculating sentiment scores: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on error


# Plot the sentiment scores
def plot_sentiment(df: pd.DataFrame, ticker: str) -> go.Figure:
    try:
        return sentimentAlgo.plot_sentiment()
    except Exception as e:
        print(f"Error plotting sentiment for {ticker}: {str(e)}")
        return go.Figure()  # Return an empty figure on error


# Get the earliest date for the news data
def get_earliest_date(df: pd.DataFrame) -> pd.Timestamp:
    try:
        date = df['Date Time'].iloc[-1]
        py_date = date.to_pydatetime()
        return py_date.replace(tzinfo=EST)
    except Exception as e:
        print(f"Error getting earliest date: {str(e)}")
        return pd.Timestamp.now()  # Return the current timestamp if there's an issue


# Plot hourly price data for the stock
def plot_hourly_price(df, ticker: str) -> go.Figure:
    try:
        fig = px.line(data_frame=df, x=df['Date Time'], y="Price", title=f"{ticker} Price")
        return fig
    except Exception as e:
        print(f"Error plotting hourly price for {ticker}: {str(e)}")
        return go.Figure()  # Return an empty figure on error


# Home route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Analyze route, triggered when the form is submitted
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        ticker = request.form['ticker'].strip().upper()

        # 1. Get news feed
        news_df = get_news(ticker)
        if news_df.empty:
            return f"No news data found for {ticker}. Please try again later."

        # 2. Calculate sentiment scores
        scored_news_df = score_news(news_df)
        if scored_news_df.empty:
            return f"Failed to calculate sentiment scores for {ticker}. Please try again later."

        # 3. Create a bar diagram for sentiment analysis
        fig_bar_sentiment = plot_sentiment(scored_news_df, ticker)
        graph_sentiment = json.dumps(fig_bar_sentiment, cls=PlotlyJSONEncoder)

        # 4. Get the earliest datetime from the news data feed
        earliest_datetime = get_earliest_date(news_df)

        # 5. Get price history for the ticker, ignore data earlier than the news feed
        price_history_df = get_price_history(ticker, earliest_datetime)
        if price_history_df.empty:
            return f"Failed to fetch price history for {ticker}. Please try again later."

        # 6. Create a linear diagram for the price history
        fig_line_price_history = plot_hourly_price(price_history_df, ticker)
        graph_price = json.dumps(fig_line_price_history, cls=PlotlyJSONEncoder)

        # 7. Make the Headline column clickable (optional)
        scored_news_df = convert_headline_to_link(scored_news_df)

        # 8. Render output
        return render_template('analysis.html', ticker=ticker, graph_price=graph_price, graph_sentiment=graph_sentiment,
                               table=scored_news_df.to_html(classes='mystyle', render_links=True, escape=False))

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return f"An error occurred during analysis: {str(e)}"


# Convert the 'title' column to clickable links
def convert_headline_to_link(df: pd.DataFrame) -> pd.DataFrame:
    df.insert(2, 'Headline', df['title + link'])
    df.drop(columns=['sentiment', 'title + link', 'title'], inplace=True, axis=1)
    return df


# Main entry point for the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81, debug=True, load_dotenv=True)