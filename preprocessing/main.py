from pathlib import Path
import numpy as np

from preprocessing.files import FileHandler, TiffHandler, H5Handler
from preprocessing.clipping import histogram

def main():
    data_path = Path("/dls/science/groups/imaging/ramona/segmentation/19505_data.tif")

    data_handle = TiffHandler(file=data_path)
    output_handler = H5Handler()

    for (handle, label) in [(data_handle, 'clipped_up_preserve')]:
        data = handle.read()
        flat = data.flatten()
        lb, ub = histogram(flat)
        data[data < lb] = np.NaN
        data[data > ub] = ub
        # set NaNs to mean
        # data = np.nan_to_num(data, nan=np.nanmean(data))
        # rescale the data to [0, 1]
        print(f'min: {np.nanmin(data)}, max: {np.nanmin(data)}')
        data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        print(f'min: {np.nanmin(data)}, max: {np.nanmax(data)}')
        # scaler = MinMaxScaler()
        # data = scaler.fit_transform(data)
        output_handler.write(
            Path().absolute(),
            data,
            label
        )