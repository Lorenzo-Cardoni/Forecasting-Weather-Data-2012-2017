import pandas as pd

# Carica il dataset
data = pd.read_csv('Weather/temperature_new.csv')

# Converti la colonna 'datetime' in un formato datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Imposta la colonna 'datetime' come indice del DataFrame
data.set_index('datetime', inplace=True)

# Seleziona solo la colonna di Los Angeles
los_angeles_pressure = data['Los Angeles']

# Resample dei dati a intervalli settimanali e calcolo della media settimanale
los_angeles_weekly_mean = los_angeles_pressure.resample('M').mean()

# Salva i risultati in un file CSV (opzionale)
los_angeles_weekly_mean.to_csv('Weather/temperature4.csv')

# Stampa le prime righe della media settimanale
print(los_angeles_weekly_mean.head())
