import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Prognozowanie cen kryptowalut ")
st.markdown("Aplikacja do analizy i prognozowania cen kryptowalut za pomocą sieci neuronowych.")

ticker = st.selectbox("Wybierz kryptowalutę:", ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD"])
period = st.selectbox("Zakres danych:", ["1y", "3y", "5y", "10y"])

@st.cache_data
def fetch_data_from_yfinance(ticker, period):
    data = yf.download(ticker, period=period, interval='1d')
    if data.empty:
        st.error("Nie udało się pobrać danych.")
        return None
    data = data[['Close']].rename(columns={'Close': 'price'})
    data.fillna(method='ffill', inplace=True)
    return data

data = fetch_data_from_yfinance(ticker, period)
if data is not None:
    st.subheader("Podstawowe statystyki")
    st.write(data.describe())

    st.subheader("Wykres ceny w czasie")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['price'], label='Cena', color='blue')
    ax.set_title(f'Cena {ticker} w czasie')
    ax.set_xlabel('Data')
    ax.set_ylabel('Cena (USD)')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    decomposition = seasonal_decompose(data['price'], model='additive', period=30)
    st.subheader("Analiza sezonowości")

    fig, axs = plt.subplots(4, 1, figsize=(14, 8))
    axs[0].plot(decomposition.observed, label='Observed', color='blue')
    axs[0].set_title('Observed')
    axs[1].plot(decomposition.trend, label='Trend', color='orange')
    axs[1].set_title('Trend')
    axs[2].plot(decomposition.seasonal, label='Seasonal', color='green')
    axs[2].set_title('Seasonal')
    axs[3].plot(decomposition.resid, label='Residual', color='red')
    axs[3].set_title('Residual')
    for ax in axs:
        ax.grid()
    plt.tight_layout()
    st.pyplot(fig)

    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data['price'].values.reshape(-1, 1))
    test_scaled = scaler.transform(test_data['price'].values.reshape(-1, 1))

    def build_dataset(dataset, time_step=7):
        X, y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X).reshape(-1, time_step, 1), np.array(y)

    time_step = 7
    X_train, y_train = build_dataset(train_scaled, time_step)
    X_test, y_test = build_dataset(test_scaled, time_step)

    if st.button("Trenuj model"):
        with st.spinner("Trening modelu..."):
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

            y_pred = model.predict(X_test)
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

            st.subheader("Wyniki predykcji")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index[train_size + time_step:], test_data['price'][time_step:], label="Prawdziwe ceny", color="blue")
            ax.plot(data.index[train_size + time_step:], y_pred, label="Prognoza", color="red", linestyle="dashed")
            ax.set_title("Porównanie prognozy i rzeczywistych cen")
            ax.legend()
            ax.grid()
            st.pyplot(fig)

            mae = mean_absolute_error(test_data['price'][time_step:], y_pred)
            mse = mean_squared_error(test_data['price'][time_step:], y_pred)
            r2 = r2_score(test_data['price'][time_step:], y_pred)

            st.write(f"**MAE:** {mae:.4f}")
            st.write(f"**MSE:** {mse:.4f}")
            st.write(f"**R² Score:** {r2:.4f}")

st.markdown("---")
st.markdown(" *Autor: [Dawid Pytliński]* | *Kod źródłowy: [GitHub](https://github.com/dawid-pytlinski/Deep-Learning-Analysis-Thesis.git)*")
