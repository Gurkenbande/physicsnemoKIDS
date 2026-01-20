from examples.weather.corrdiff.datasets.cwb import get_zarr_dataset

data_path = "/home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/Data/cwa_dataset_3months.zarr"
ds = get_zarr_dataset(data_path=data_path, all_times=False)


print(f"Input Channels are {ds.input_channels()}")
print(f"Output Channels are{ds.output_channels()}")
print(f"Image Shape is{ds.image_shape()}")
print(f"Longitude is{ds.longitude()}")
print(f"Latitude is{ds.latitude()}")

