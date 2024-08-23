import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import datetime as dt
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
import numpy as np
from statsmodels.tsa.stattools import adfuller


# Caricamento del dataset
file = 'Weather/temperature_new.csv'
df = pd.read_csv(file)

# Imposta la colonna datetime come indice
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Controlla e gestisci i valori NaN o infiniti
df['Los Angeles'].replace([np.inf, -np.inf], np.nan, inplace=True)  # Sostituisci inf con NaN
df['Los Angeles'].interpolate(method='time', inplace=True)  # Interpolazione dei valori mancanti

# Funzione per ACF e PACF
def ACF_PACF(df, path_name):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df[1:], lags=40, ax=ax1)  # Valore diff=NaN ignorato
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(df[1:], lags=40, ax=ax2)
    plt.savefig(path_name)
    plt.show()


# ADF test per la stazionarietà
def ADF_test(df):
    result2 = adfuller(df.values)
    print('ADF Statistic: %f' % result2[0])
    print('p-value: %f' % result2[1])

def ARIMA_model(p, d, q, df):
    model = ARIMA(df.values, order=(p, d, q))
    ax = plt.gca()
    results = model.fit()
    plt.plot(df)
    plt.plot(pd.DataFrame(results.fittedvalues, columns=['Los Angeles']).set_index(df.index), color='red')
    ax.legend(['Temperature', 'Forecast'])
    plt.savefig('ARIMA/Predictions', bbox_inches='tight')
    ax.set_xlim(dt.datetime(2017, 1, 1), dt.datetime(2017, 11, 30))
    plt.show()
    print(results.summary())

    # residual error
    residuals = pd.DataFrame(results.resid).set_index(df.index)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.savefig('ARIMA/Residual_Error', bbox_inches='tight')
    plt.show()

    results.plot_diagnostics(figsize=(12, 8))
    plt.savefig('ARIMA/Diagnostics', bbox_inches='tight')
    plt.show()


# Funzione per addestrare il modello ARIMA
def train_ARIMA(p, d, q, df):
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    model = ARIMA(train_data, order=(p, d, q))
    arima_model = model.fit()
    
    pred_uc = arima_model.get_forecast(steps=len(test_data))
    pred_ci = pred_uc.conf_int()
    
    # Grafico dei risultati
    ax = train_data.plot(color='b', label='Train')
    test_data.plot(color='r', label='Test', ax=ax)
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast', style='k--')
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.05)
    
    plt.legend()
    plt.show()
    
    return pred_uc, test_data

# Esegui il test di stazionarietà e il training del modello
ADF_test(df['Los Angeles'])

# Plot ACF e PACF
ACF_PACF(df['Los Angeles'], 'ARIMA')

ARIMA_model(1, 0, 0, df['Los Angeles'])

# Allenamento del modello ARIMA
p, d, q = 1, 0, 0  # Ordini dell'ARIMA da regolare in base ai risultati di ACF/PACF
pred_uc, test_data = train_ARIMA(p, d, q, df['Los Angeles'])

# Calcolo delle metriche di errore
predicted = pred_uc.predicted_mean
mape = met.mean_absolute_percentage_error(test_data, predicted)
sqe = met.mean_squared_error(test_data, predicted)
mae = met.mean_absolute_error(test_data, predicted)
r2 = met.r2_score(test_data, predicted)

print(f"MAPE: {mape}")
print(f"MSE: {sqe}")
print(f"MAE: {mae}")
print(f"R^2: {r2}")

