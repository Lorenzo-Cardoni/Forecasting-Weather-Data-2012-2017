import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
import numpy as np

# Caricamento del dataset
file = 'Weather/temperature4.csv'
df = pd.read_csv(file)

# Imposta la colonna datetime come indice
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Controlla e gestisci i valori NaN o infiniti
df['Los Angeles'].replace([np.inf, -np.inf], np.nan, inplace=True)  # Sostituisci inf con NaN
df['Los Angeles'].interpolate(method='time', inplace=True)  # Interpolazione dei valori mancanti

stag = 12 # stagionalità della serie temporale

# Funzione per addestrare il modello ARIMA usando auto_arima
def auto_arima_model(df):
    # Usa auto_arima per selezionare i migliori parametri (senza stagionalità)
    model = pm.auto_arima(df, 
                          seasonal=True,    # Disabilita la stagionalità
                          m=stag,   
                          stepwise=True,        # Ricerca dei parametri passo-passo
                          suppress_warnings=True, 
                          trace=True)           # Mostra le informazioni del processo
    
    # Fitta il modello sui dati di addestramento
    model.fit(df)
    
    print(model.summary())
    
    return model


# Funzione per il forecasting usando il modello auto_arima
def forecast_auto_arima(model, train_data, test_data):
    # Predizione
    forecast = model.predict(n_periods=len(test_data), return_conf_int=True)
    pred_mean = forecast[0]
    pred_ci = forecast[1]
    
    # Grafico dei risultati
    ax = train_data.plot(color='b', label='Train')
    test_data.plot(color='r', label='Test', ax=ax)
    pd.Series(pred_mean, index=test_data.index).plot(ax=ax, label='Forecast', style='k--')
    
    ax.fill_between(test_data.index, pred_ci[:, 0], pred_ci[:, 1], color='k', alpha=.15)
    plt.legend()
    plt.savefig("ARIMA\sarimax4.png")
    plt.show()
    
    return pred_mean


# Funzione per il forecasting usando il modello auto_arima
def forecast_auto_arima2(model, data, period):
    # Predizione
    forecast = model.predict(period, return_conf_int=True)
    pred_mean = forecast[0]
    pred_ci = forecast[1]
    
    date_rng = pd.date_range(start=data.index[-1], periods=period, freq='M')

    # Grafico dei risultati
    ax = data.plot(color='b', label='data')
    pd.Series(pred_mean, index=date_rng).plot(ax=ax, label='Forecast', style='k--')
    
    ax.fill_between(date_rng, pred_ci[:, 0], pred_ci[:, 1], color='k', alpha=.15)
    plt.legend()
    plt.savefig("ARIMA\sarimax_outsample4.png")
    plt.show()
    
    return pred_mean

# Dividi i dati in train e test set (80% train, 20% test)
train_data, test_data = train_test_split(df['Los Angeles'], test_size=0.2, shuffle=False)

# Allenamento del modello con auto_arima
model = auto_arima_model(train_data)

model.plot_diagnostics()
plt.show()

# Esegui il forecasting sui dati di test
predicted = forecast_auto_arima(model, train_data, test_data)

# Esegui il forecasting sui dati di test
predicted2 = forecast_auto_arima2(model, df['Los Angeles'], 36)

# Calcolo delle metriche di errore
mape = met.mean_absolute_percentage_error(test_data, predicted)
mse = met.mean_squared_error(test_data, predicted)
mae = met.mean_absolute_error(test_data, predicted)
r2 = met.r2_score(test_data, predicted)

print(f"MAPE: {mape}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R^2: {r2}")




