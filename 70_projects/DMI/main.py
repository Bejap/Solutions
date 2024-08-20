import requests

def fetch_data(api_url, api_key, limit):
    all_data = []
    offset = 0
    while len(all_data) < limit:
        response = requests.get(
            f'{api_url}?bbox-crs=https%3A%2F%2Fwww.opengis.net%2Fdef%2Fcrs%2FOGC%2F1.3%2FCRS84&api-key={api_key}&offset={offset}'
        )
        data = response.json()
        features = data.get('features', [])
        all_data.extend(features)
        if len(features) < 1000:  # Assuming each page contains 1000 records or fewer
            break
        offset += len(features)
    return all_data

api_url = 'https://dmigw.govcloud.dk/v2/lightningdata/collections/sensordata/items'
api_key = '1eb30e5a-fd28-4df4-85ad-0d14b2ad81b9'
record_limit = 2000

features = fetch_data(api_url, api_key, record_limit)

# Process features
coordinates = [feature['geometry']['coordinates'] for feature in features]
created = [feature['properties']['created'] for feature in features]

# Separate the coordinates into longitude (x) and latitude (y)
longitudes = [coord[0] for coord in coordinates]
latitudes = [coord[1] for coord in coordinates]

# Find min and max values
min_longitude = min(longitudes)
max_longitude = max(longitudes)
min_latitude = min(latitudes)
max_latitude = max(latitudes)

# Find coordinates corresponding to min and max values
min_longitude_coord = coordinates[longitudes.index(min_longitude)]
max_longitude_coord = coordinates[longitudes.index(max_longitude)]
min_latitude_coord = coordinates[latitudes.index(min_latitude)]
max_latitude_coord = coordinates[latitudes.index(max_latitude)]

print(f"Minimum Longitude (x): {min_longitude} found at {min_longitude_coord}")
print(f"Maximum Longitude (x): {max_longitude} found at {max_longitude_coord}")
print(f"Minimum Latitude (y): {min_latitude} found at {min_latitude_coord}")
print(f"Maximum Latitude (y): {max_latitude} found at {max_latitude_coord}")

for i in range(len(created)):
    print(i)
