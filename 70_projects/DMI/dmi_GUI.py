import tkinter as tk
from tkinter import filedialog
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SimpleGISApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple GIS with Tkinter")

        # Set up the main frame
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Add a button to load shapefiles
        self.load_button = tk.Button(self.frame, text="Load Shapefile", command=self.load_shapefile)
        self.load_button.pack(side=tk.TOP)

        # Set up the Matplotlib figure
        self.fig, self.ax = plt.subplots()

    def load_shapefile(self):
        # Open a file dialog to select the shapefile
        shapefile_path = filedialog.askopenfilename(title="Open Shapefile", filetypes=[("Shapefiles", "*.shp")])
        if shapefile_path:
            # Load the shapefile using GeoPandas
            gdf = gpd.read_file(shapefile_path)

            # Clear the existing plot
            self.ax.clear()

            # Plot the data
            gdf.plot(ax=self.ax)

            # Embed the plot in the Tkinter canvas
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleGISApp(root)
    root.mainloop()

