# DMI's dokumentation om hvordan man henter lynnedslag data:
# https://opendatadocs.dmi.govcloud.dk/APIs/Lightning_Data_API

import requests
import numpy as np
from datetime import datetime

class DataCollection:

    def __init__(self, my_limit, query_limit):
        _base_url = 'https://dmigw.govcloud.dk/v2/lightningdata/collections/observation/items'
        self.api_url = _base_url
        self.api_key = '1eb30e5a-fd28-4df4-85ad-0d14b2ad81b9'
        self.my_limit = my_limit
        self.query_limit = query_limit

    def fetch_data(self, start_date, end_date, south_west_corner, north_east_corner):
        longitude_min, latitude_min = south_west_corner
        longitude_max, latitude_max = north_east_corner
        all_data = []
        offset = 0
        while True:
            url = (f'{self.api_url}'
                   f'?bbox={longitude_min},{latitude_min},{longitude_max},{latitude_max}'
                   f'&datetime={convert_date(start_date)}/{convert_date(end_date)}'
                   f'&api-key={self.api_key}'
                   f'&limit={self.query_limit}'
                   f'&offset={offset}')
            print(url)
            response = requests.get(url)
            data_ = response.json()
            records = data_.get('features', [])
            if not records:
                print("Found no new records.")
                break
            print("Found", len(records), "new records.")
            # for record in records:
            #     print(record)
            all_data.extend(records)
            print("Found", len(all_data), "records so far.")
            if len(records) < self.query_limit:
                print("These are less new records than the query's limit, therefore are there no more records to request.")
                break
            offset += self.query_limit
            print("Requesting more data. Offset now set to", offset)
            if len(all_data) >= self.my_limit:
                print("Stopping request, because I found already at least", self.my_limit, "records.")
                break
        return all_data[:self.my_limit]


def convert_date(date):
    year, month, day = date
    formatted_date = datetime(year, month, day)
    return formatted_date.strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    start_date = (2000, 1, 1)  # year, month, day
    end_date = (2024, 12, 31)
    south_west_corner = (12.4408, 55.6241)  # longitude, latitude
    north_east_corner = (12.4539, 55.6311)
    data = DataCollection(2000, 100)
    records = data.fetch_data(start_date, end_date, south_west_corner, north_east_corner)
    ids = [record['id'] for record in records]
    print("Found", len(np.unique(ids)), "unique ids.")
    dates = [record['properties']['observed'] for record in records]
    for date in sorted(dates):
        print(date)


if __name__ == '__main__':
    main()
