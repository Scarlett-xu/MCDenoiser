"""
MIT License

Copyright (c) [2025] [Crystal Collaborators]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import OpenEXR
import Imath
import numpy as np
import os
import cv2

import matplotlib.pyplot as plt


def save_float_array_as_png(arr: np.array, filename: str, clamp_scale: (float,float)):
    """
    Save a single-channel float array as a PNG image with 4 channels.

    vbnet
    复制代码
    Steps:
    1. Clamp the array values to the range [0, clamp_scale].
    2. Normalize the clamped array to the range [0, 1].
    3. Convert the normalized array to 0-255 np.uint8 type.
    4. Replicate the single channel to create a 4-channel image.
    5. Save the output image as a PNG file.

    Parameters:
    arr         : Input single-channel float array.
    filename    : File name to save the PNG image (e.g., 'output.png').
    clamp_scale : Upper bound for clamping. Values greater than this will be set to clamp_scale.
    """
    # Step 1: Clamp the values in the array to the range [0, clamp_scale]
    arr_clamped = np.clip(arr, clamp_scale[0], clamp_scale[1])
    # Step 2: Normalize the clamped array to the range [0, 1]
    arr_normalized = (arr_clamped -clamp_scale[0])  / (clamp_scale[1]-clamp_scale[0])  # Now the data is in [0, 1]
    # Step 3: Convert the normalized array to the range [0, 255] and cast to np.uint8
    img_uint8 = (arr_normalized * 255).astype(np.uint8)
    # Step 4: Replicate the single channel to create a 4-channel image
    # If the original shape is (height, width), the new shape becomes (height, width, 4)
    img_4ch = np.stack([img_uint8] * 4, axis=-1)
    # Step 5: Save the image as a PNG file using OpenCV
    cv2.imwrite(filename, img_4ch)

def save_3_float_arrays_as_png(arr1: np.array, arr2: np.array, arr3: np.array, filename: str, clamp_scale: (float,float)):
    """
    Save three single-channel float arrays as a 4-channel PNG image.

    sql
    复制代码
    Steps:
    1. Clamp each array to the range [0, clamp_scale].
    2. Normalize each clamped array to the range [0, 1].
    3. Convert each normalized array to the range [0, 255] and cast to np.uint8.
    4. Create a fourth channel filled with 255.
    5. Stack the three channels and the fourth full 255 channel to form a image.
    6. Save the image as a PNG file.

    Parameters:
    arr1, arr2, arr3 : Input single-channel float arrays.
    filename         : File name for saving the PNG image (e.g., 'output.png').
    clamp_scale      : Upper bound for clamping. Values greater than this will be set to clamp_scale.
    """
    # Step 1: Clamp the input arrays to [0, clamp_scale]
    arr1_clamped = np.clip(arr1, clamp_scale[0], clamp_scale[1])
    arr2_clamped = np.clip(arr2, clamp_scale[0], clamp_scale[1])
    arr3_clamped = np.clip(arr3, clamp_scale[0], clamp_scale[1])

    # Step 2: Normalize each array to the range [0, 1]
    arr1_normalized = (arr1_clamped -clamp_scale[0])  / (clamp_scale[1]-clamp_scale[0])
    arr2_normalized = (arr2_clamped -clamp_scale[0])  / (clamp_scale[1]-clamp_scale[0])
    arr3_normalized = (arr3_clamped -clamp_scale[0])  / (clamp_scale[1]-clamp_scale[0])

    # Step 3: Convert each normalized array to np.uint8 (range [0, 255])
    arr1_uint8 = (arr1_normalized * 255).astype(np.uint8)
    arr2_uint8 = (arr2_normalized * 255).astype(np.uint8)
    arr3_uint8 = (arr3_normalized * 255).astype(np.uint8)

    # Ensure input arrays have the same dimensions; get shape from one of them.
    height, width = arr1_uint8.shape

    # Step 4: Create a fourth channel with all values set to 255
    channel_four = np.full((height, width), 255, dtype=np.uint8)

    # Step 5: Stack the three processed channels and the constant channel along the last dimension
    # The output image shape will be (height, width, 4)
    output_image = np.stack([arr1_uint8, arr2_uint8, arr3_uint8, channel_four], axis=-1)

    # Step 6: Save the resulting image as a PNG file
    cv2.imwrite(filename, output_image)
    print(f"Image saved as {filename}")

def loadExr(input_exr_path):
    """
    Loads an OpenEXR (.exr) file and extracts channel data into a dictionary.

    Args:
    - input_exr_path (str): Path to the input EXR file.

    Returns:
    - channel_data_dict (dict): Dictionary containing channel names as keys and numpy arrays
      representing channel data as values.
    """
    exr_file = OpenEXR.InputFile(input_exr_path)
    # Get the header information of the EXR file
    header = exr_file.header()
    width = header['dataWindow'].max.x - header['dataWindow'].min.x + 1
    height = header['dataWindow'].max.y - header['dataWindow'].min.y + 1

    # Get a list of channel names
    channel_names = list(header["channels"].keys())

    # Dictionary to store channel data
    channel_data_dict = {}

    # Traverse each channel
    for channel_name in channel_names:
        # Read channel data
        channel_data = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))

        # Convert channel data to numpy array (optional, depending on usage)
        channel_data_np = np.frombuffer(channel_data, dtype=np.float32)
        channel_data_np = np.reshape(channel_data_np, (height, width))  # Reshape to match image dimensions

        # Store channel data in the dictionary
        channel_data_dict[channel_name] = channel_data_np

    return width, height, channel_data_dict

def Display_Exr_exposured(input_exr_path, exposure=0.5):
    """
    Loads an OpenEXR (.exr) file, performs tone mapping on each channel data based on exposure,
    and displays the mapped channels in a grid using matplotlib.

    Args:
    - input_exr_path (str): Path to the input EXR file.
    - exposure (float): Exposure value used for tone mapping adjustment.

    Returns:
    - None

    This function loads the EXR file using the loadExr function, computes the inverse exposure
    value for tone mapping, and then iterates through each channel's data. It applies a clamp
    operation to ensure the data is within valid range, performs tone mapping to adjust brightness,
    and converts the adjusted data to displayable grayscale images. The mapped images are displayed
    in a grid layout using matplotlib subplots.
    """

    exr_file = OpenEXR.InputFile(input_exr_path)
    # 读取文件的头信息
    header = exr_file.header()
    # 打印头信息
    print("EXR File Header Information:")
    for key, value in header.items():
        print(f"{key}: {value}")

    width, height, data = loadExr(input_exr_path)

    invExposure = 1.0 / (1.0 - exposure)

    cols = 4
    num_channels = len(data)
    # Round up
    rows = (num_channels + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

    for idx, (channel_name, channel_data) in enumerate(data.items()):
        print(f"Channel Name: {channel_name}")
        print(f"Shape of Channel Data: {channel_data.shape}\n")
        # Clamp
        channel_data = np.clip(channel_data, 0, None)
        # Tone mapping
        channel_data = 1.0 - np.exp(-channel_data * invExposure)

        mapped_data = (channel_data * 255).astype(np.uint8)

        row = idx // cols
        col = idx % cols

        if rows == 1:
            axs[col].imshow(mapped_data, cmap='gray')
            axs[col].set_title(channel_name)
            axs[col].axis('off')
        else:
            axs[row, col].imshow(mapped_data, cmap='gray')
            axs[row, col].set_title(channel_name)
            axs[row, col].axis('off')

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.4)
    plt.tight_layout()
    plt.show()

def Display_Exr_normalized(input_exr_path):
    """
    Load the exr file and normalize each channel based on its maximum and minimum values

    Args:
    - input_exr_path (str): Path to the input EXR file.

    Returns:
    - None

    This function loads the EXR file using the loadExr function, computes the inverse exposure
    value for tone mapping, and then iterates through each channel's data. It applies a clamp
    operation to ensure the data is within valid range, performs tone mapping to adjust brightness,
    and converts the adjusted data to displayable grayscale images. The mapped images are displayed
    in a grid layout using matplotlib subplots.
    """

    exr_file = OpenEXR.InputFile(input_exr_path)
    # 读取文件的头信息
    header = exr_file.header()
    # 打印头信息
    print("EXR File Header Information:")
    for key, value in header.items():
        print(f"{key}: {value}")


    width, height, data = loadExr(input_exr_path)

    cols = 4
    num_channels = len(data)
    # Round up
    rows = (num_channels + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

    for idx, (channel_name, channel_data) in enumerate(data.items()):
        print(f"Channel Name: {channel_name}")
        print(f"Shape of Channel Data: {channel_data.shape}")

        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        print(f"Min: {min_val}, Max: {max_val}\n")

        interval = (max_val - min_val)
        if 0 == interval:
            channel_data = np.ones_like(channel_data)
        else:
            channel_data = (channel_data - min_val) / interval

        mapped_data = (channel_data * 255).astype(np.uint8)

        row = idx // cols
        col = idx % cols

        if rows == 1:
            axs[col].imshow(mapped_data, cmap='gray')
            axs[col].set_title(channel_name)
            axs[col].axis('off')
        else:
            axs[row, col].imshow(mapped_data, cmap='gray')
            axs[row, col].set_title(channel_name)
            axs[row, col].axis('off')

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.4)
    plt.tight_layout()
    plt.show()


def str_to_matrix(mat_str) -> np.ndarray:

    content = mat_str
    if (content.startswith(b'[')) :
        content = content[1:]
    if (content.endswith(b']')) :
        content = content[0:-1]

    items = content.split(b',')
    if len(items) != 16:
        raise KeyError(f"The number is not 16, cannot get 4X4 matrix")

    try:
        # Convert each number to float
        numbers = [float(item) for item in items]
    except ValueError:
        raise KeyError(f"An error occurred while converting numbers")

    matrix = np.array(numbers).reshape(4, 4)
    return matrix


def divide_images(array1: np.array, array2: np.array) -> np.array:
    """
    Divide image array1 by image array2 while handling near-zero divisors.

    Steps:
    1. For each element in array2, if it is less than 0.001, set it to 0.
    2. For positions where array2 is 0, set the corresponding output to 0.
    3. For positions where array2 is greater than or equal to 0.001, compute array1 / array2.

    Parameters:
    array1 : numpy array of float, the numerator image.
    array2 : numpy array of float, the denominator image.

    Returns:
    A new numpy array representing the division result.
    """

    # Ensure array1 and array2 have the same shape
    if array1.shape != array2.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Initialize the result array with zeros
    result = np.zeros_like(array1)

    # Create a mask where array2 is safe to use as the divisor
    safe_mask = array2 >= 0.001

    # Perform division only on safe positions
    np.divide(array1, array2, out=result, where=safe_mask)

    return result

def loadExr_with_transforms_nullscattering(
        input_exr_path,
        check_channel = True,
        reprojection_test=False,
        denoising=True,
        generatePng=False):
    """
    Loads an OpenEXR (.exr) file and extracts channel data into a dictionary.

    Args:
    - input_exr_path (str): Path to the input EXR file.

    Returns:
    - channel_data_dict (dict): Dictionary containing channel names as keys and numpy arrays
      representing channel data as values.
    """
    exr_file = OpenEXR.InputFile(input_exr_path)
    # Get the header information of the EXR file
    header = exr_file.header()
    width = header['dataWindow'].max.x - header['dataWindow'].min.x + 1
    height = header['dataWindow'].max.y - header['dataWindow'].min.y + 1

    information = {}
    for key, value in header.items():
        if (key == 'Camera2Screen'):
            information['Camera2Screen'] = str_to_matrix(value)
        elif (key == 'VolumeToWorld'):
            information['VolumeToWorld'] = str_to_matrix(value)
        elif (key == 'World2Camera'):
            information['World2Camera'] = str_to_matrix(value)
        elif (key == 'Screen2Raster'):
            information['Screen2Raster'] = str_to_matrix(value)
    if 'Camera2Screen' not in information:
        raise KeyError(f"There is no Camera2Screen matrix")
    if 'VolumeToWorld' not in information:
        raise KeyError(f"There is no VolumeToWorld matrix")
    if 'World2Camera' not in information:
        raise KeyError(f"There is no World2Camera matrix")
    if 'Screen2Raster' not in information:
        raise KeyError(f"There is no Screen2Raster matrix")

    Volume2Camera = np.dot(information['World2Camera'], information['VolumeToWorld'])
    Camera2Raster = np.dot(information['Screen2Raster'], information['Camera2Screen'])
    Volume2Raster = np.dot(Camera2Raster, Volume2Camera)
    information['Volume2Raster'] = Volume2Raster

    # Get a list of channel names
    channel_names = list(header["channels"].keys())

    # Dictionary to store channel data
    channel_data_dict = {}
    # Traverse each channel
    for channel_name in channel_names:
        # Read channel data
        channel_data = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))

        # Convert channel data to numpy array (optional, depending on usage)
        channel_data_np = np.frombuffer(channel_data, dtype=np.float32)
        channel_data_np = np.reshape(channel_data_np, (height, width))  # Reshape to match image dimensions

        # Store channel data in the dictionary
        channel_data_dict[channel_name] = channel_data_np

    if check_channel:
        # check channel data
        if 'Alpha' not in channel_data_dict:
            raise KeyError(f"There is no Alpha channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Alpha'])) > 0:
            raise KeyError(f"Alpha has nan data")

        if 'Sdf_R' not in channel_data_dict:
            raise KeyError(f"There is no Sdf_R channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Sdf_R'])) > 0:
            raise KeyError(f"Sdf_R has nan data")
        if 'Sdf_G' not in channel_data_dict:
            raise KeyError(f"There is no Sdf_G channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Sdf_G'])) > 0:
            raise KeyError(f"Sdf_G has nan data")
        if 'Sdf_B' not in channel_data_dict:
            raise KeyError(f"There is no Sdf_B channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Sdf_B'])) > 0:
            raise KeyError(f"Sdf_B has nan data")

        if 'SingleRadi_R' not in channel_data_dict:
            raise KeyError(f"There is no SingleRadi_R channel")
        if np.count_nonzero(np.isnan(channel_data_dict['SingleRadi_R'])) > 0:
            raise KeyError(f"SingleRadi_R has nan data")
        if 'SingleRadi_G' not in channel_data_dict:
            raise KeyError(f"There is no SingleRadi_G channel")
        if np.count_nonzero(np.isnan(channel_data_dict['SingleRadi_G'])) > 0:
            raise KeyError(f"SingleRadi_G has nan data")
        if 'SingleRadi_B' not in channel_data_dict:
            raise KeyError(f"There is no SingleRadi_B channel")
        if np.count_nonzero(np.isnan(channel_data_dict['SingleRadi_B'])) > 0:
            raise KeyError(f"SingleRadi_B has nan data")

        if 'Illumin_R' not in channel_data_dict:
            raise KeyError(f"There is no Illumin_R channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Illumin_R'])) > 0:
            raise KeyError(f"Illumin_R has nan data")
        if 'Illumin_G' not in channel_data_dict:
            raise KeyError(f"There is no Illumin_G channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Illumin_G'])) > 0:
            raise KeyError(f"Illumin_G has nan data")
        if 'Illumin_B' not in channel_data_dict:
            raise KeyError(f"There is no Illumin_B channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Illumin_B'])) > 0:
            raise KeyError(f"Illumin_B has nan data")

        if 'MultiRadi_R' not in channel_data_dict:
            raise KeyError(f"There is no MultiRadi_R channel")
        if np.count_nonzero(np.isnan(channel_data_dict['MultiRadi_R'])) > 0:
            raise KeyError(f"MultiRadi_R has nan data")
        if 'MultiRadi_G' not in channel_data_dict:
            raise KeyError(f"There is no MultiRadi_G channel")
        if np.count_nonzero(np.isnan(channel_data_dict['MultiRadi_G'])) > 0:
            raise KeyError(f"MultiRadi_G has nan data")
        if 'MultiRadi_B' not in channel_data_dict:
            raise KeyError(f"There is no MultiRadi_B channel")
        if np.count_nonzero(np.isnan(channel_data_dict['MultiRadi_B'])) > 0:
            raise KeyError(f"MultiRadi_B has nan data")

        if 'Pos_X' not in channel_data_dict:
            raise KeyError(f"There is no Pos_X channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Pos_X'])) > 0:
            raise KeyError(f"Pos_X has nan data")
        if 'Pos_Y' not in channel_data_dict:
            raise KeyError(f"There is no Pos_Y channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Pos_Y'])) > 0:
            raise KeyError(f"Pos_Y has nan data")
        if 'Pos_Z' not in channel_data_dict:
            raise KeyError(f"There is no Pos_Z channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Pos_Z'])) > 0:
            raise KeyError(f"Pos_Z has nan data")

        if 'Normal_X' not in channel_data_dict:
            raise KeyError(f"There is no Normal_X channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Normal_X'])) > 0:
            raise KeyError(f"Normal_X has nan data")
        if 'Normal_Y' not in channel_data_dict:
            raise KeyError(f"There is no Normal_Y channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Normal_Y'])) > 0:
            raise KeyError(f"Normal_Y has nan data")
        if 'Normal_Z' not in channel_data_dict:
            raise KeyError(f"There is no Normal_Z channel")
        if np.count_nonzero(np.isnan(channel_data_dict['Normal_Z'])) > 0:
            raise KeyError(f"Normal_Z has nan data")

    # reprojection testing
    if reprojection_test:
        for i in range(0, 100):

            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            while (channel_data_dict['Alpha'][row, col] == 0.0):
                row = np.random.randint(0, height)
                col = np.random.randint(0, width)

            x_pos = channel_data_dict['Pos_X'][row, col]
            y_pos = channel_data_dict['Pos_Y'][row, col]
            z_pos = channel_data_dict['Pos_Z'][row, col]
            point_homogeneous = np.array([x_pos, y_pos, z_pos, 1])
            transformed_point = Volume2Raster @ point_homogeneous
            if transformed_point[3] != 0:
                spatial_point = transformed_point[:3] / transformed_point[3]
            else:
                raise KeyError(f"transformed_point calculate error")
            #print(f"point:{col},{row}, reprojection: {spatial_point[0]},{spatial_point[1]},{spatial_point[2]}")

            if (abs(spatial_point[0] - col) > 2 or abs(spatial_point[1] - row) > 2) :
                print(f"Pos: {x_pos},{y_pos},{z_pos};alpha:{channel_data_dict['Alpha'][row, col]}")

    if generatePng:
        current_dir = os.getcwd()
        folder_name = "Save-python"
        folder_path = os.path.join(current_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        save_float_array_as_png(channel_data_dict['Alpha'], 'Save-python/Alpha.png', (0, 1))
        save_3_float_arrays_as_png(
            channel_data_dict['Sdf_R'], channel_data_dict['Sdf_G'], channel_data_dict['Sdf_B'],
            'Save-python/Sdf.png', (0, 0.1))

    if denoising:
        channel_data_dict['Alpha'] = cv2.ximgproc.guidedFilter(
            channel_data_dict['Alpha'], channel_data_dict['Alpha'], 8, 0.01)

        channel_data_dict['Sdf_R'] = cv2.ximgproc.guidedFilter(
            channel_data_dict['Sdf_R'], channel_data_dict['Sdf_R'], 4, 0.00001)
        channel_data_dict['Sdf_G'] = cv2.ximgproc.guidedFilter(
            channel_data_dict['Sdf_G'], channel_data_dict['Sdf_G'], 4, 0.00001)
        channel_data_dict['Sdf_B'] = cv2.ximgproc.guidedFilter(
            channel_data_dict['Sdf_B'], channel_data_dict['Sdf_B'], 4, 0.00001)

        if generatePng:
            save_float_array_as_png(channel_data_dict['Alpha'], 'Save-python/Denoised_Alpha.png', (0, 1))
            save_3_float_arrays_as_png(
                channel_data_dict['Sdf_R'], channel_data_dict['Sdf_G'], channel_data_dict['Sdf_B'],
                'Save-python/Denoised_Sdf.png', (0, 0.1))

    # generate Q component
    channel_data_dict['Q_Comp_R'] = divide_images(channel_data_dict['SingleRadi_R'], channel_data_dict['Sdf_R'])
    channel_data_dict['Q_Comp_G'] = divide_images(channel_data_dict['SingleRadi_G'], channel_data_dict['Sdf_G'])
    channel_data_dict['Q_Comp_B'] = divide_images(channel_data_dict['SingleRadi_B'], channel_data_dict['Sdf_B'])
    if generatePng:
        save_3_float_arrays_as_png(
            channel_data_dict['Q_Comp_R'], channel_data_dict['Q_Comp_G'], channel_data_dict['Q_Comp_B'],
            'Save-python/Q_Comp.png', (0, 20))


    return width, height, channel_data_dict, information













