import requests

def fetch_data(api_url, api_key, limit):
    all_data = []
    offset = 0
    while len(all_data) < limit:
        response = requests.get(
            f'{api_url}?bbox=7,54,16,58&api-key={api_key}&offset={offset}'
        )
        data = response.json()
        features = data.get('features', [])
        all_data.extend(features)
        if len(features) < 1000:  # Assuming each page contains 1000 records or fewer
            break
        offset += len(features)
    return all_data

api_url = 'https://dmigw.govcloud.dk/v2/lightningdata/collections/observation/items'
api_key = '1eb30e5a-fd28-4df4-85ad-0d14b2ad81b9'
limit = 2000

features = fetch_data(api_url, api_key, limit)


created = [feature['properties']['created'] for feature in features]
u = []
i = 0
for list in range(len(created)):
    i = created[list]
    listing = created[list]
    print(listing)
    var = listing[:4]
    print(var)
    j = 0
    u.append(var)


print(u)
print(len(u))

