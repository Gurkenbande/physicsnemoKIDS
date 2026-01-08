import xarray as xr
import matplotlib.pyplot as plt

# Pfad anpassen (lokal ODER Cluster)
DATA_PATH = "hrrr_mini_train.nc"

# Dataset laden
ds = xr.open_dataset(DATA_PATH)

print(" Dataset:")
print(ds)

print("\n Dimensionen:")
print(ds.dims)

print("\n Variablen:")
for var in ds.data_vars:
    print("-", var)

# Beispiel: erste Variable plotten
var_name = list(ds.data_vars.keys())[0]
sample = ds[var_name].isel(time=0)

plt.figure(figsize=(6, 5))
plt.imshow(sample, origin="lower")
plt.colorbar()
plt.title(var_name)
plt.show()

