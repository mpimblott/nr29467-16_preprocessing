
from pathlib import Path


import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import exposure

from preprocessing.files.file_handler import FileHandler
from preprocessing.files import NPYHandler, TiffHandler, H5Handler

def adaptive_histogram_equalization(
    data: np.ndarray,
    kernel_size: np.ndarray = np.array([32, 32, 32]),
    clip_limit: float = 0.9
) -> np.ndarray:
    data = exposure.equalize_adapthist(data, kernel_size=kernel_size, clip_limit=clip_limit)
    return data

def denoise(
    data: np.ndarray,
):
    # sample = data[data.shape[0]//2, :, :]
    # plt.imshow(sample, cmap='gray')
    # plt.savefig('denoise_original_sample.png')
    # plt.close()
    # sigma_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # # create subplots
    # fig, axs = plt.subplots(1, len(sigma_vals), figsize=(1.5*len(sigma_vals),3))
    # for i, sigma in enumerate(sigma_vals):
    #     x = scipy.ndimage.gaussian_filter(
    #         sample,
    #         sigma=sigma,
    #     )
    #     axs[i].imshow(x, cmap='gray')
    #     axs[i].set_title(f'sigma={sigma}')
    #     axs[i].axis('off')

    # # save the image
    # plt.savefig('denoise_gaussian.png')
    # plt.close()
    data = scipy.ndimage.gaussian_filter(
        data,
        sigma=1.5,
    )
    return data

def apply_denoise():
    data_path = Path('/dls/science/users/jig77871/projects/projection2/outputs/clipped_up_preserve.h5')
    data_handle = H5Handler(file=data_path)
    data = data_handle.read()
    # data = denoise(data)
    output_handler = H5Handler()
    output_handler.write(Path().absolute(), data, 'denoised')