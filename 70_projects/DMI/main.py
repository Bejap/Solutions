import dmi_GUI as dmg
import dmi_get_data as dgd
import webbrowser as wb
import numpy as np


def get_data():
    start_date = (2000, 1, 1)
    end_date = (2003, 12, 31)
    south_west_corner = (11, 54)
    north_east_corner = (14, 56)
    data = dgd.DataCollection(20000, 1000)
    records = data.fetch_data(start_date, end_date, south_west_corner, north_east_corner)
    ids = [record['id'] for record in records]
    print("found", len(np.unique(ids)), "unique strokes")
    dates = [record['properties']['observed'] for record in records]
    for date in sorted(dates):
        print(date)
    location = [record['geometry']['coordinates'] for record in records]
    return location

def transform_data(sublist):
    new_list = sorted(sublist, reverse=True)
    return new_list


if __name__ == '__main__':
    DK_map = dmg.DanishMap()
    data_list = get_data()
    lon_lat = [transform_data(data) for data in data_list]
    DK_map.create_markers(lon_lat)
    wb.open('map.html')



