import requests





# Fetch the data from the API
response = requests.get('https://dmigw.govcloud.dk/v2/lightningdata/collections/sensordata/items?bbox-crs=https%3A%2F%2Fwww.opengis.net%2Fdef%2Fcrs%2FOGC%2F1.3%2FCRS84&api-key=1eb30e5a-fd28-4df4-85ad-0d14b2ad81b9')

# Parse the response as JSON
data = response.json()

# Extract the coordinates
coordinates = [feature['geometry']['coordinates'] for feature in data['features']]
created = [feature['properties']['created'] for feature in data['features']]

# Separate the coordinates into longitude (x) and latitude (y)
longitudes = [coord[0] for coord in coordinates]
latitudes = [coord[1] for coord in coordinates]

min_longitude = min(longitudes)
max_longitude = max(longitudes)
min_latitude = min(latitudes)
max_latitude = max(latitudes)

min_longitude_coord = coordinates[longitudes.index(min_longitude)]
max_longitude_coord = coordinates[longitudes.index(max_longitude)]
min_latitude_coord = coordinates[latitudes.index(min_latitude)]
max_latitude_coord = coordinates[latitudes.index(max_latitude)]

print(f"Minimum Longitude (x): {min_longitude} found at {min_longitude_coord}")
print(f"Maximum Longitude (x): {max_longitude} found at {max_longitude_coord}")
print(f"Minimum Latitude (y): {min_latitude} found at {min_latitude_coord}")
print(f"Maximum Latitude (y): {max_latitude} found at {max_latitude_coord}")

for i in created:
    print(i)

