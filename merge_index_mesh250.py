import pandas as pd
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np

import pandas as pd

mesh_250 = pd.read_csv("../final_code/data/mesh_quarter_features.csv").dropna()
############################

#mesh_250 = pd.read_csv("mesh_quarter_features.csv").dropna()
price_index = pd.read_csv("../final_code/data/mesh_quarterly_price_index.csv")

price_index["Mesh250m"] = price_index["Mesh250m"].astype(str)
mesh_250["Mesh250m"] = mesh_250["Mesh250m"].astype(str)

price_index["PeriodKey"] = price_index["PeriodKey"].astype(str)
mesh_250["PeriodKey"] = mesh_250["PeriodKey"].astype(str)

df = pd.merge(mesh_250, price_index, how="left", on=["Mesh250m", "PeriodKey"])

df.to_csv("mesh_250_w_price_index.csv")

half_side_m = 125  # 250m / 2

def make_square(lat, lon, half_m=125):
    dlat = half_m / 111320
    dlon = half_m / (111320 * np.cos(np.radians(lat)))
    return Polygon([
        (lon - dlon, lat - dlat),
        (lon + dlon, lat - dlat),
        (lon + dlon, lat + dlat),
        (lon - dlon, lat + dlat)
    ])

df["geometry"] = df.apply(lambda r: make_square(r["Latitude"], r["Longitude"]), axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
gdf.to_file("mesh250_all_quarters.geojson", driver="GeoJSON")
print("✅ Ready:", len(gdf), "records across", gdf["PeriodKey"].nunique(), "quarters")