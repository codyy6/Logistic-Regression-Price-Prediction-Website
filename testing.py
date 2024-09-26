import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

@st.cache_data
def fetch_and_prepare_data():
    # Fetch data and prepare features
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(period="max")
    df = df.reset_index()
    df = df.sort_values('Date')
    
    # Create features
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df['Price_Change'] = df['Close'].diff()
    df['High_Low_Diff'] = df['High'] - df['Low']
    df['Open_Close_Diff'] = df['Close'] - df['Open']
    df['Volume_Change'] = df['Volume'].diff()
    df['Adj_Close_Diff'] = df['Close'].diff()
    
    # Create target variable
    df['Target'] = (df['Price_Change'] > 0).astype(int)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

@st.cache_resource
def train_model(df):
    # Select features and target
    X = df[['Days', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'High_Low_Diff', 'Open_Close_Diff', 'Volume_Change', 'Adj_Close_Diff']]
    y = df['Target']
    
    # Split the data and scale features
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def predict_future_prices(df, model, scaler, days=30):
    last_day = df['Days'].max()
    last_price = df['Close'].iloc[-1]
    future_days = np.array(range(last_day + 1, last_day + days + 1))
    
    # Create future features
    future_features = pd.DataFrame({
        'Days': future_days,
        'Open': [last_price] * days,
        'High': [last_price] * days,
        'Low': [last_price] * days,
        'Close': [last_price] * days,
        'Volume': [df['Volume'].mean()] * days,
        'Price_Change': [0] * days,
        'High_Low_Diff': [0] * days,
        'Open_Close_Diff': [0] * days,
        'Volume_Change': [0] * days,
        'Adj_Close_Diff': [0] * days
    })
    
    future_features_scaled = scaler.transform(future_features)
    future_probabilities = model.predict_proba(future_features_scaled)[:, 1]
    
    # Calculate predicted prices
    predicted_changes = (future_probabilities - 0.5) * df['Price_Change'].std()
    predicted_prices = [last_price]
    for change in predicted_changes:
        predicted_prices.append(predicted_prices[-1] + change)
    predicted_prices = predicted_prices[1:]
    
    # Generate dates for predictions
    last_date = df['Date'].max()
    future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days + 1)]
    
    return list(zip(future_dates, predicted_prices))

def main():
    st.title('BTC-USD Price Prediction')
    
    # Fetch and prepare data
    with st.spinner('Fetching and preparing data...'):
        df = fetch_and_prepare_data()
    
    # Train model
    with st.spinner('Training model...'):
        model, scaler = train_model(df)
    
    # Make predictions
    with st.spinner('Making predictions...'):
        predictions = predict_future_prices(df, model, scaler)
    
    # Display predictions
    st.subheader('Predicted Prices for the Next 30 Days')
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Historical Data'
    ))
    
    # Add predicted data
    fig.add_trace(go.Scatter(
        x=[p[0] for p in predictions],
        y=[p[1] for p in predictions],
        mode='lines+markers',
        name='Predicted Prices',
        line=dict(color='red')
    ))
    
    # Update layout
    fig.update_layout(
        title='BTC-USD Historical and Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified'
    )
    
    # Display the plot
    st.plotly_chart(fig)
    
    # Display predictions in a table
    st.subheader('Predicted Prices Table')
    df_predictions = pd.DataFrame(predictions, columns=['Date', 'Predicted Price'])
    df_predictions['Predicted Price'] = df_predictions['Predicted Price'].round(2)
    st.dataframe(df_predictions)

if __name__ == '__main__':
    main()