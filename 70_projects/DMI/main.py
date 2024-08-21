import requests
import numpy as np

class Data_Collection:

    def __init__(self, limit, year_to_collect):
        _base_url = 'https://dmigw.govcloud.dk/v2/lightningdata/collections/observation/items'
        self.api_url = _base_url
        self.api_key = '1eb30e5a-fd28-4df4-85ad-0d14b2ad81b9'
        self.limit = limit
        self.year_to_collect = year_to_collect

    def fetch_data(self):
        date_precision = '-01-01T00:00:00%2B02:00'
        all_data = []
        offset = 0
        while len(all_data) < self.limit:
            response = requests.get(
            f'{self.api_url}?bbox=7,54,16,58&datetime={str(self.year_to_collect)}{date_precision}/{str(self.year_to_collect + 1)}{date_precision}&api-key={self.api_key}&offset={offset}'
        )
            data = response.json()
            features = data.get('features', [])
            if not features:
                break
            all_data.extend(features)
            offset += 1000
            if len(all_data) >= self.limit:
                break

        return all_data[:self.limit]


year_to_collect = 2010
limit = 20000

data = Data_Collection(limit, year_to_collect)
features = data.fetch_data()

coordinates = [feature['geometry']['coordinates'] for feature in features]
years = [feature['properties']['observed'] for feature in features]
id = [feature['id'] for feature in features]
ID = np.unique(id)

print(len(ID))

# u = []
# for i in range(len(years)):
#     j = str([years[i], coordinates[i]])
#     u.append(j)
#
# unique_data = set(u)
# unique_id = set(id)
#
# for entry in unique_data:
#     print(entry)
#
# print(f'Unique entries: {len(unique_data)}')
