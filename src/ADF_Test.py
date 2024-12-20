import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Carica i dati dai file CSV
temperature_df = pd.read_csv('Weather/temperature4.csv', parse_dates=['datetime'])

# Seleziona i dati per Los Angeles
los_angeles_temperature = temperature_df[['datetime', 'Los Angeles']].set_index('datetime')

# Funzione per eseguire l'ADF test e stampare i risultati
def adf_test(series, series_name):
    print(f"Results of ADF Test for {series_name}:")
    adf_result = adfuller(series.dropna())
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    print(f'ADF Statistic: {adf_statistic}')
    print(f'p-value: {p_value}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')
    print('\n')

# Esegui l'ADF test per ciascuna serie temporale
adf_test(los_angeles_temperature['Los Angeles'], 'temperature')


# Funzione per decomporre la serie temporale
def decompose_series(series, series_name, model='additive'):
    result = seasonal_decompose(series.dropna(), model=model, period=12)  
    result.plot()
    plt.suptitle(f'{series_name} - Decomposition', fontsize=16)
    plt.savefig("plots/adf_test4.png")
    plt.show()

# Decomponi e visualizza per ciascuna serie temporale
decompose_series(los_angeles_temperature['Los Angeles'], 'Temperature')