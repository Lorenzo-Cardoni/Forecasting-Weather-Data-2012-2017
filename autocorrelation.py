import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Carica i dati dai file CSV
humidity_df = pd.read_csv('Weather/humidity_new.csv', parse_dates=['datetime'])
pressure_df = pd.read_csv('Weather/pressure_new.csv', parse_dates=['datetime'])
temperature_df = pd.read_csv('Weather/temperature_new.csv', parse_dates=['datetime'])

# Seleziona i dati per Los Angeles
los_angeles_humidity = humidity_df[['datetime', 'Los Angeles']].set_index('datetime')
los_angeles_pressure = pressure_df[['datetime', 'Los Angeles']].set_index('datetime')
los_angeles_temperature = temperature_df[['datetime', 'Los Angeles']].set_index('datetime')

#dati differenziati al primo ordine
humidity_diff1 = los_angeles_humidity['Los Angeles'].diff().dropna()
pressure_diff1 = los_angeles_pressure['Los Angeles'].diff().dropna()
temperature_diff1 = los_angeles_temperature['Los Angeles'].diff().dropna()



plt.figure(figsize=(14, 10))

# Umidità
plt.subplot(3, 2, 1)
plot_acf(humidity_diff1, ax=plt.gca())
plt.title('ACF of Differenced Humidity')

plt.subplot(3, 2, 2)
plot_pacf(humidity_diff1, ax=plt.gca())
plt.title('PACF of Differenced Humidity')

# Pressione
plt.subplot(3, 2, 3)
plot_acf(pressure_diff1, ax=plt.gca())
plt.title('ACF of Differenced Pressure')

plt.subplot(3, 2, 4)
plot_pacf(pressure_diff1, ax=plt.gca())
plt.title('PACF of Differenced Pressure')

# Temperatura
plt.subplot(3, 2, 5)
plot_acf(temperature_diff1, ax=plt.gca())
plt.title('ACF of Differenced Temperature')

plt.subplot(3, 2, 6)
plot_pacf(temperature_diff1, ax=plt.gca())
plt.title('PACF of Differenced Temperature')

plt.tight_layout()
plt.show()
