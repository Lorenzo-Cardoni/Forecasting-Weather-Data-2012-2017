import pandas as pd
import matplotlib.pyplot as plt

# Carica i dati dai file CSV
humidity_df = pd.read_csv('Weather/humidity_new.csv', parse_dates=['datetime'])
pressure_df = pd.read_csv('Weather/pressure_new.csv', parse_dates=['datetime'])
temperature_df = pd.read_csv('Weather/temperature_new.csv', parse_dates=['datetime'])

# Seleziona i dati per Los Angeles
los_angeles_humidity = humidity_df[['datetime', 'Los Angeles']]
los_angeles_pressure = pressure_df[['datetime', 'Los Angeles']]
los_angeles_temperature = temperature_df[['datetime', 'Los Angeles']]

# Imposta la colonna 'datetime' come indice
los_angeles_humidity.set_index('datetime', inplace=True)
los_angeles_pressure.set_index('datetime', inplace=True)
los_angeles_temperature.set_index('datetime', inplace=True)

# Crea una figura e tre subplot
plt.figure(figsize=(14, 10))

# Grafico dell'umidità
plt.subplot(3, 1, 1)
plt.plot(los_angeles_humidity, label='Humidity', color='blue')
plt.title('Humidity in Los Angeles')
plt.ylabel('Humidity (%)')
plt.grid(True)

# Grafico della pressione
plt.subplot(3, 1, 2)
plt.plot(los_angeles_pressure, label='Pressure', color='green')
plt.title('Pressure in Los Angeles')
plt.ylabel('Pressure (hPa)')
plt.grid(True)

# Grafico della temperatura
plt.subplot(3, 1, 3)
plt.plot(los_angeles_temperature, label='Temperature', color='red')
plt.title('Temperature in Los Angeles')
plt.ylabel('Temperature (°C)')
plt.grid(True)

# Migliora la spaziatura tra i subplot
plt.tight_layout()

# Mostra i grafici
plt.show()
