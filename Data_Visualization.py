import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Caricare il file CSV
file_path = 'Weather/city_attributes_new.csv'  # Sostituisci con il percorso corretto del file CSV
cities_df = pd.read_csv(file_path)

# Definire gli offset per le città per evitare la sovrapposizione
offsets = {
    'Seattle': (0.5, -0.5),
    'Portland': (0.5, -0.5),
    'San Francisco': (0.5, -0.5),
    'Los Angeles': (0.5, -0.5),
    'San Diego': (0.5, -0.5),
    'Las Vegas': (0.5, 0.5),
    'Phoenix': (0.5, 0.5),
    'Albuquerque': (0.5, 0.5),
    'Denver': (0.5, 0.5),
    'San Antonio': (-5.5, 0.5),
    'Dallas': (-0.5, 0.5),
    'Houston': (-0.5, 0.5),
    'Minneapolis': (0.5, -0.5),
    'Chicago': (0.5, -0.5),
    'Detroit': (0.5, -0.5),
    'Indianapolis': (0.5, -0.5),
    'Saint Louis': (0.5, -0.5),
    'Kansas City': (-0.5, 1.2),
    'Nashville': (0.5, -0.5),
    'Atlanta': (0.5, -0.5),
    'Charlotte': (0.5, -0.5),
    'Jacksonville': (0.5, -0.5),
    'Miami': (0.5, -0.5),
    'Pittsburgh': (0.5, -0.5),
    'Toronto': (0.5, -0.5),
    'Philadelphia': (0.5, -0.5),
    'New York': (0.5, -0.5),
    'Montreal': (0.5, -0.5),
    'Boston': (0.5, -0.5),
    # Aggiungi altre città se necessario
}

# Creazione della figura e dell'asse con una proiezione geografica
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Aggiungere caratteristiche della mappa come linee costiere e confini
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)

# Aggiungere le città alla mappa
for index, row in cities_df.iterrows():
    ax.plot(row['Longitude'], row['Latitude'], marker='o', color='red', markersize=5, transform=ccrs.Geodetic())
    offset_lon, offset_lat = offsets.get(row['City'], (0.5, 0.5))  # Offset di default se la città non è nel dizionario
    ax.text(row['Longitude'] + offset_lon, row['Latitude'] + offset_lat, row['City'], transform=ccrs.Geodetic())

# Impostare i limiti della mappa (opzionale, centrato sugli Stati Uniti e Canada)
ax.set_extent([-130, -60, 20, 60], crs=ccrs.PlateCarree())

plt.title('Città in Nord America')
plt.show()
