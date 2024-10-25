import requests
import pandas as pd
from datetime import datetime
import folium


class DataCollection:

    def __init__(self, filename, center, start_date, end_date, south_west_corner, north_east_corner, my_limit, query_limit):
        _base_url = 'https://dmigw.govcloud.dk/v2/lightningdata/collections/observation/items'
        self.api_url = _base_url
        self.api_key = '1eb30e5a-fd28-4df4-85ad-0d14b2ad81b9'
        self.filename = filename
        self.center = center
        self.start_date = start_date
        self.end_date = end_date
        self.south_west_corner = south_west_corner
        self.north_east_corner = north_east_corner
        self.my_limit = my_limit
        self.query_limit = query_limit

    def fetch_data(self):
        west, south = self.south_west_corner
        east, north = self.north_east_corner
        all_data = []
        offset = 0
        while True:
            url = (f'{self.api_url}'
                   f'?bbox={west},{south},{east},{north}'
                   f'&datetime={convert_date(self.start_date)}/{convert_date(self.end_date)}'
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

    def dmi2csv(self):
        records = self.fetch_data()
        lightningstrokes2earth = [LightningStroke(record['geometry']['coordinates'][0],
                                                  record['geometry']['coordinates'][1],
                                                  datetime.strptime(record['properties']['observed'], "%Y-%m-%dT%H:%M:%S.%fZ"))
                                  for record in records if float(record['properties']['type']) < 2.0]  # Type 2 = cloud2cloud lightning excluded!

        sorted_strokes = sorted(lightningstrokes2earth)
        print(f'Data contains {len(sorted_strokes)} records of cloud to earth lightning strokes.')
        # for lightningstroke in sorted_strokes:
        #     print(lightningstroke)
        df = pd.DataFrame([(stroke.lon, stroke.lat, stroke.datetime) for stroke in sorted_strokes],
                          columns=['Longitude', 'Latitude', 'Datetime'])
        # print(df)
        df.to_csv(self.filename, index=False)
        print("DataFrame has been saved as lightning_filename")

    def csv2map(self):
        df = pd.read_csv(self.filename)
        # print(df)

        m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=14, control_scale=True)  # Create a map centered around the specified coordinates
        west, south = self.south_west_corner
        east, north = self.north_east_corner

        kw = {
            "color": "darkgreen",
            "line_cap": "round",
            "fill": True,
            "fill_color": "lightgrey",
            "weight": 2,
            # "popup": "",
            # "tooltip": "",
        }
        folium.Rectangle(bounds=[[south, west], [north, east]], line_join="round", dash_array="4, 8", **kw).add_to(m)
        # icon = folium.Icon(prefix="fa", color="red", icon_color="#1ABC9C", icon="home")
        icon = folium.Icon(prefix="fa", color="red", icon="home")
        folium.Marker([self.center[1], self.center[0]], icon=icon, tooltip=f"Center of area with data").add_to(m)

        for idx, row in df.iterrows():  # Add markers for each row in the DataFrame
            icon = folium.Icon(prefix="fa", color="darkblue", icon_color="#eeee44", icon="bolt")
            folium.Marker([row['Latitude'], row['Longitude']], icon=icon, tooltip=f"Lightning Strike at {row['Datetime']}").add_to(m)

        m.save("map.html")


class LightningStroke:

    def __init__(self, lon, lat, date_time):
        self.lon = lon
        self.lat = lat
        self.datetime = date_time

    def __repr__(self):
        return f'{self.__class__.__name__}({self.lon=:.4f}, {self.lat=:.4f}, {self.datetime.strftime("%d.%m.%Y, %H:%M:%S")})'

    def __lt__(self, other):
        return self.datetime < other.datetime


def convert_date(date):
    year, month, day = date
    formatted_date = datetime(year, month, day)
    return formatted_date.strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    center = (12.27, 55.647)
    lon_width = 0.050
    lat_width = 0.035
    start_date = (2000, 1, 1)
    end_date = (2099, 12, 31)
    filename = 'lightning_strokes.csv'
    lon_center, lat_center = center  # center of map

    south_west_corner = (lon_center - lon_width / 2, lat_center - lat_width / 2)
    north_east_corner = (lon_center + lon_width / 2, lat_center + lat_width / 2)
    data = DataCollection(filename, center, start_date, end_date, south_west_corner, north_east_corner, 2000, 1000)
    data.dmi2csv()
    data.csv2map()


if __name__ == '__main__':
    main()
