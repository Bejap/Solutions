import folium as fl


class DanishMap:
    def __init__(self):
        min_lat, max_lat = 52, 60
        min_lon, max_lon = 6, 15

        self.DK_map = fl.Map(max_bounds=True,
                             location=(55.58, 11.46),
                             zoom_start=7,
                             min_lat=min_lat,
                             max_lat=max_lat,
                             min_lon=min_lon,
                             max_lon=max_lon)
        self.DK_map.save('map.html')

    def create_markers(self, lon_lat_list):
        for i in range(len(lon_lat_list)):
            fl.Circle(radius=25,
                      color="black",
                      location=lon_lat_list[i],
                      tooltip="Click me",
                      popup=f"Lightning stroke here at some date",
                      fill_color="black",
                      opacity=1.2,
                      fill_opacity=0.5,
                      fill=False,
                      weight=1).add_to(self.DK_map)
        self.DK_map.save('map.html')
