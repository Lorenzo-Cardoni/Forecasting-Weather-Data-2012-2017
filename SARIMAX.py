import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime as dt
import sklearn.metrics as met
from ARIMA import ADF_test
from ARIMA import ACF_PACF
from statsmodels.tsa.seasonal import seasonal_decompose

# Caricamento del dataset
file = 'Weather/temperature_new.csv'
df = pd.read_csv(file)

# Imposta la colonna datetime come indice
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Seleziona la colonna della temperatura, ad esempio Los Angeles
temperature = df['Los Angeles']

# Decomposizione stagionale
seasonal_diff = seasonal_decompose(temperature, model='additive', extrapolate_trend='freq').seasonal

# Funzione per la previsione successiva
def next_prediction(p, d, q, P, D, Q, m, df):
    # Costruzione del modello SARIMAX
    mod = SARIMAX(df.values, order=(p, d, q), seasonal_order=(P, D, Q, m), trend='c')
    res = mod.fit(disp=0)

    # Previsione
    forecast_steps = 50
    fcast = res.get_forecast(steps=forecast_steps).summary_frame()
    forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=forecast_steps, freq='D')
    fcast.index = forecast_index

    # Plot
    fig, ax = plt.subplots(figsize=(15, 5))
    df.plot(ax=ax, label='Dati storici', color='skyblue')
    fcast['mean'].plot(ax=ax, style='k--', label='Previsione')
    ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1)

    plt.xlabel('Data')
    plt.ylabel('Temperatura')
    plt.title('Previsione SARIMAX')
    plt.legend()
    plt.savefig('SARIMAX/next_prediction.png', bbox_inches='tight')
    plt.show()

# Modello SARIMAX e visualizzazione
def SARIMAX_model(p, d, q, P, D, Q, m, df):
    model = SARIMAX(df.values, order=(p, d, q), seasonal_order=(P, D, Q, m))
    results = model.fit()

    # Plot
    ax = plt.gca()
    plt.plot(df)
    plt.plot(pd.DataFrame(results.fittedvalues, columns=['Predicted']).set_index(df.index), color='red')
    ax.legend(['Temperatura', 'Previsione'])
    plt.savefig('SARIMAX/Predictions.png', bbox_inches='tight')
    ax.set_xlim(dt.datetime(2017, 1, 1), dt.datetime(2018, 1, 9))
    plt.show()
    print(results.summary())

    # Residui
    residuals = pd.DataFrame(results.resid)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    residuals.plot(title="Residui", ax=ax[0])
    residuals.plot(kind='kde', title='Densità', ax=ax[1])
    plt.savefig('SARIMAX/Residual_Error.png', bbox_inches='tight')
    plt.show()

    results.plot_diagnostics(figsize=(12, 8))
    plt.savefig('SARIMAX/Diagnostics.png', bbox_inches='tight')
    plt.show()

# Funzione per addestrare e testare il modello SARIMAX
def train_SARIMAX(p, d, q, P, D, Q, m, df):
    # Dividi i dati in training e test
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    index_list = list(df.index)
    train_len = int(len(index_list) * 0.8)
    date_split = str(index_list[train_len + 1])
    train_data = df[:date_split]
    test_data = df[date_split:]
    
    ax = df.plot(color='b', label='Train')
    df.loc[date_split:].plot(color='r', label='Test', ax=ax)

    # Modello SARIMAX su dati di training
    model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, m))
    sarimax_model = model.fit()

    # Previsione
    pred_uc = sarimax_model.get_forecast(steps=len(test_data))
    pred_ci = pred_uc.conf_int()

    pd.DataFrame(pred_uc.predicted_mean).set_index(pd.DatetimeIndex(test_data.index))['predicted_mean'].plot(ax=ax, label='Previsione', style='k--')
    ax.fill_between(pred_ci.set_index(pd.DatetimeIndex(test_data.index)).index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.05)

    ax.set_xlabel('Data')
    ax.set_ylabel('Temperatura')
    ax.set_xlim(dt.datetime(2017, 1, 1), dt.datetime(2018, 1, 9))

    plt.legend()
    plt.savefig('SARIMAX/Forecasting.png', bbox_inches='tight')
    plt.show()

    return pred_uc, test_data

# Test di stazionarietà e ACF/PACF
ADF_test(seasonal_diff)
ACF_PACF(seasonal_diff, 'SARIMAX/ACF_PACF.png')

# Esecuzione del modello e training
SARIMAX_model(1, 0, 2, 1, 0, 1, 7, temperature)
pred_uc, test_data = train_SARIMAX(1, 0, 2, 1, 0, 1, 7, temperature)

# Previsione successiva
next_prediction(1, 0, 1, 40, 0, 20, 7, temperature)

# Calcolo degli errori
predicted = pred_uc.predicted_mean
mape = met.mean_absolute_percentage_error(test_data, predicted)
sqe = met.mean_squared_error(test_data.squeeze(), predicted)
mae = met.mean_absolute_error(test_data, predicted)
r2 = met.r2_score(test_data, predicted)
print(f'MAPE: {mape:.4f}')
print(f'MSE: {sqe:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R2: {r2:.4f}')
