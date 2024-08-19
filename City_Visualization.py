import pandas as pd
import folium

# Carica il file CSV
df = pd.read_csv('Weather/city_attributes_new.csv')

# Crea una mappa centrata intorno alla media delle latitudini e longitudini
mappa = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=5)

# Aggiungi i marker per ogni città
for index, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['City']}, {row['Country']}"
    ).add_to(mappa)

# Salva la mappa come file HTML
mappa.save('mappa_citta.html')
