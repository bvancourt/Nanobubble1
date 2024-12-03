"""
This file contains helper functions used by NpRegionSelector.py.
"""

import tkinter as tk
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal



def dot_image(color, diameter):
    radius = (
        diameter + 3
    ) / 2
    coordinate_profile = np.arange(-np.ceil(radius) + 1, np.ceil(radius))
    x, y = np.meshgrid(coordinate_profile, coordinate_profile)
    d_squared = x**2 + y**2 + 1

    alpha = (
        np.clip((radius**2 / d_squared - 1) * 255 * np.sqrt(radius), 0, 255)
    ).astype(np.uint8)
    red = np.ones(x.shape, dtype=np.uint8) * color[0]
    green = np.ones(x.shape, dtype=np.uint8) * color[1]
    blue = np.ones(x.shape, dtype=np.uint8) * color[2]

    return np.dstack([red, green, blue, alpha])


def alpha_drop(
    dest_image, new_image, location=[0, 0], mode="top_left"
):  # should be rewritten to use alpha_recolor() (below).
    if mode == "center":
        new_to_dest_i = (
            location[0]
            - new_image.shape[0] // 2
            + np.arange(new_image.shape[0])
        )
        new_to_dest_j = (
            location[1]
            - new_image.shape[1] // 2
            + np.arange(new_image.shape[1])
        )

    if mode == "top_left":
        new_to_dest_i = location[0] + np.arange(new_image.shape[0])
        new_to_dest_j = location[1] + np.arange(new_image.shape[1])

    i, j = np.meshgrid(new_to_dest_j, new_to_dest_i)
    valid = (
        (i >= 0)
        & (j >= 0)
        & (i < dest_image.shape[0])
        & (j < dest_image.shape[1])
    )
    alpha = new_image[:, :, 3]
    for k in range(dest_image.shape[2]):
        dest_image[i[valid], j[valid], k] = (
            dest_image[i[valid], j[valid], k] * (255 - alpha[valid])
            + alpha[valid] * new_image[:, :, k][valid]
        )

    return dest_image


def alpha_recolor(base_im, alpha_im):
    return np.dstack(
        (
            base_im[:, :, 0] * (255 - alpha_im[:, :, 3]) / 255
            + alpha_im[:, :, 0] * alpha_im[:, :, 3] / 255,
            base_im[:, :, 1] * (255 - alpha_im[:, :, 3]) / 255
            + alpha_im[:, :, 1] * alpha_im[:, :, 3] / 255,
            base_im[:, :, 2] * (255 - alpha_im[:, :, 3]) / 255
            + alpha_im[:, :, 2] * alpha_im[:, :, 3] / 255,
        )
    ).astype(np.uint8)


def resamp_grid(source, tlc, scale, shape, pixel_aspect_ratio=1):
    j, i = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    return (
        -pixel_aspect_ratio * tlc[0] + pixel_aspect_ratio * i / scale,
        -tlc[1] + j / scale,
    )


def autozoom_grid(source, out_shape, method="pad", pixel_aspect_ratio=1):
    source_ar = (
        source.shape[0] / source.shape[1] / pixel_aspect_ratio
    )  # aspect ration of the image arrays... not the individual pixels.
    output_ar = (
        out_shape[0] / out_shape[1]
    )  # aspect ration of the image arrays... not the individual pixels.
    if method == "pad":
        if source_ar >= output_ar:  # dimension 0 defines zoom amount
            scale = out_shape[0] / source.shape[0] * pixel_aspect_ratio
            tlc = [0, int(out_shape[1] / scale - source.shape[1]) // 2]
            return resamp_grid(
                source,
                tlc,
                scale,
                out_shape,
                pixel_aspect_ratio=pixel_aspect_ratio,
            )
        else:  # dimension 1 defines zoom out
            scale = out_shape[1] / source.shape[1]
            tlc = [
                int(out_shape[0] / scale - source.shape[0] / pixel_aspect_ratio)
                // 2,
                0,
            ]
            return resamp_grid(
                source,
                tlc,
                scale,
                out_shape,
                pixel_aspect_ratio=pixel_aspect_ratio,
            )
    if method == "fill":
        if source_ar < output_ar:  # dimension 0 defines zoom amount
            scale = out_shape[0] / source.shape[0] * pixel_aspect_ratio
            tlc = [0, int(out_shape[1] / scale - source.shape[1]) // 2]
            return resamp_grid(
                source,
                tlc,
                scale,
                out_shape,
                pixel_aspect_ratio=pixel_aspect_ratio,
            )
        else:  # dimension 1 defines zoom out
            scale = out_shape[1] / source.shape[1]
            tlc = [
                int(out_shape[0] / scale - source.shape[0] / pixel_aspect_ratio)
                // 2,
                0,
            ]
            return resamp_grid(
                source,
                tlc,
                scale,
                out_shape,
                pixel_aspect_ratio=pixel_aspect_ratio,
            )
    else:
        raise ValueError(
            "resize_grid() recieved unrecognized apect ratio mismatch method: "
            + f"{method}"
        )


def autozoom_to_source_coords(
    x_screen, y_screen, source, out_shape, method="pad", pixel_aspect_ratio=1
):
    source_ar = source.shape[0] / source.shape[1] / pixel_aspect_ratio
    output_ar = out_shape[0] / out_shape[1]
    if method == "pad":
        if source_ar >= output_ar:  # dimension 0 defines zoom amount
            scale = out_shape[0] / source.shape[0] * pixel_aspect_ratio
            tlc = [0, int((out_shape[1] / scale - source.shape[1]) / 2)]
            return (-tlc[0] + x_screen / scale) * pixel_aspect_ratio, -tlc[
                1
            ] + y_screen / scale
        else:  # dimension 1 defines zoom out
            scale = out_shape[1] / source.shape[1]
            tlc = [
                int(
                    (
                        out_shape[0] / scale
                        - source.shape[0] / pixel_aspect_ratio
                    )
                    / 2
                ),
                0,
            ]
            return (-tlc[0] + x_screen / scale) * pixel_aspect_ratio, -tlc[
                1
            ] + y_screen / scale
    if method == "fill":  # Have not confirmed that this method is fully working
        if source_ar < output_ar:  # dimension 0 defines zoom amount
            scale = out_shape[0] / source.shape[0] * pixel_aspect_ratio
            tlc = [0, int((out_shape[1] / scale - source.shape[1]) / 2)]
            return (-tlc[0] + x_screen / scale) * pixel_aspect_ratio, -tlc[
                1
            ] + y_screen / scale
        else:  # dimension 1 defines zoom out
            scale = out_shape[1] / source.shape[1]
            tlc = [
                int(
                    (
                        out_shape[0] / scale
                        - source.shape[0] / pixel_aspect_ratio
                    )
                    / 2
                ),
                0,
            ]
            return (-tlc[0] + x_screen / scale) * pixel_aspect_ratio, -tlc[
                1
            ] + y_screen / scale
    else:
        print(
            f"autozoom_to_source_coords() recieved unrecognized apect ratio "
            + f"mismatch method: {method}"
        )


def source_to_autozoom_coords(
    x_source, y_source, source, out_shape, method="pad", pixel_aspect_ratio=1
):
    source_ar = source.shape[0] / source.shape[1] / pixel_aspect_ratio
    output_ar = out_shape[0] / out_shape[1]
    if method == "pad":
        if source_ar >= output_ar:  # dimension 0 defines zoom amount
            scale = out_shape[0] / source.shape[0] * pixel_aspect_ratio
            tlc = [0, int((out_shape[1] / scale - source.shape[1]) / 2)]
            return (x_source + tlc[0]) * scale / pixel_aspect_ratio, (
                y_source + tlc[1]
            ) * scale
        else:  # dimension 1 defines zoom out
            scale = out_shape[1] / source.shape[1]
            tlc = [
                int(
                    (
                        out_shape[0] / scale
                        - source.shape[0] / pixel_aspect_ratio
                    )
                    / 2
                ),
                0,
            ]
            return (x_source / pixel_aspect_ratio + tlc[0]) * scale, (
                y_source + tlc[1]
            ) * scale
    if method == "fill":  # Have not confirmed that this method is fully working
        if source_ar < output_ar:  # dimension 0 defines zoom amount
            scale = out_shape[0] / source.shape[0] * pixel_aspect_ratio
            tlc = [0, int((out_shape[1] / scale - source.shape[1]) / 2)]
            return (x_source + tlc[0]) * scale / pixel_aspect_ratio, (
                y_source + tlc[1]
            ) * scale
        else:  # dimension 1 defines zoom out
            scale = out_shape[1] / source.shape[1]
            tlc = [
                int(
                    (
                        out_shape[0] / scale
                        - source.shape[0] / pixel_aspect_ratio
                    )
                    / 2
                ),
                0,
            ]
            return (x_source / pixel_aspect_ratio + tlc[0]) * scale, (
                y_source + tlc[1]
            ) * scale
    else:
        print(
            "autozoom_to_source_coords() recieved unrecognized apect ratio "
            + f"mismatch method: {method}"
        )


def im_resamp(source, x, y, method="nearest", default_color=[111, 111, 111]):
    # ideas: add an averaging method based on many average of 'nearest' method 
    #   images and 'automatic' method, that choses one of the others.
    if (x.shape == y.shape) & (source.shape[2] == len(default_color)):
        if method == "nearest":
            x_int = np.round(x).astype(int)
            y_int = np.round(y).astype(int)
            coords_inside = (
                (x_int >= 0)
                & (y_int >= 0)
                & (x_int < source.shape[0])
                & (y_int < source.shape[1])
            )
            im_out = np.zeros(
                [x.shape[0], x.shape[1], source.shape[2]], dtype=source.dtype
            )
            for i_chan in range(source.shape[2]):
                im_out[:, :, i_chan][coords_inside] = source[
                    x_int[coords_inside], y_int[coords_inside], i_chan
                ]
                im_out[:, :, i_chan][~coords_inside] = default_color[i_chan]
            return im_out

        elif method == "linear":
            im_out = np.zeros(
                [x.shape[0], x.shape[1], source.shape[2]], dtype=source.dtype
            )
            x_floor = np.floor(x).astype(int)
            y_floor = np.floor(y).astype(int)
            coords_inside = (
                (x_floor >= 1)
                & (y_floor >= 1)
                & (x_floor <= source.shape[0] - 2)
                & (y_floor <= source.shape[1] - 2)
            )
            im_out = np.zeros(
                [x.shape[0], x.shape[1], source.shape[2]], dtype=np.float32
            )
            for dx in [0, 1]:
                for dy in [0, 1]:
                    w = (1 - np.abs(dx + x_floor - x)) * (
                        1 - np.abs(dy + y_floor - y)
                    ).astype(np.float32)
                    for i_chan in range(source.shape[2]):
                        im_out[:, :, i_chan][coords_inside] += (
                            source[
                                x_floor[coords_inside] + dx,
                                y_floor[coords_inside] + dy,
                                i_chan,
                            ]
                            * w[coords_inside]
                        )
            for i_chan in range(source.shape[2]):
                im_out[:, :, i_chan][~coords_inside] = default_color[i_chan]
            return im_out.astype(source.dtype)
        else:
            print(f"im_resamp() recieved unknown method: {method}.")
    else:
        print(
            f"im_resamp() recieved invalid coordinates: x.shape={x.shape},"
            + " y.shape={y.shape}, {source.shape[2]}-channel source, "
            + "{len(default_color)}-channel default."
        )


def np_image_to_tk(image):
    height, width = image.shape[:2]
    if image.dtype == np.uint8:
        data = (
            f"P6 {width} {height} 255 ".encode()
            + image.astype(np.uint8).tobytes()
        )
    elif image.dtype in [np.float16, np.float32, np.float16]:
        if (np.max(image[~np.isnan(image)]) <= 1) and (
            np.min(image[~np.isnan(image)]) >= 0
        ):
            # In this case, assume that 0 should be black and 1 should be white.
            int_image = (image * 255).astype(np.uint8)
        else:
            int_image = (
                255
                * (image - np.nanmin(image))
                / (np.nanmax(image) - np.nanmin(image))
            ).astype(np.uint8)
        data = (
            f"P6 {width} {height} 255 ".encode()
            + int_image.astype(np.uint8).tobytes()
        )
    else:
        print(f"Image data type {image.dtype} not supported.")
    return tk.PhotoImage(width=width, height=height, data=data, format="PPM")


def point_in_triangle(x, y, corners, count_exact_edge="triangle_xor"):
    output = np.zeros(x.shape, dtype=int)

    u1 = corners[1][0] - corners[0][0]
    u2 = corners[2][0] - corners[0][0]
    v1 = corners[1][1] - corners[0][1]
    v2 = corners[2][1] - corners[0][1]
    turn_direction = np.sign(u1 * v2 - u2 * v1)
    for i in range(3):
        active_points = [corners[i], corners[i - 1]]
        u1 = active_points[1][0] - active_points[0][0]
        u2 = x - active_points[0][0]
        v1 = active_points[1][1] - active_points[0][1]
        v2 = y - active_points[0][1]
        # output += np.sign(u1*v2-u2*v1)*turn_direction
        if count_exact_edge == "never":
            output += (2 * ((u1 * v2 > u2 * v1) - 0.5)).astype(
                np.int32
            ) * turn_direction
        elif (count_exact_edge == "triangle_xor") and (i == 2):
            output += (2 * ((u1 * v2 > u2 * v1) - 0.5)).astype(
                np.int32
            ) * turn_direction
        else:
            output += (2 * ((u1 * v2 >= u2 * v1) - 0.5)).astype(
                np.int32
            ) * turn_direction

    return output == -3


# Binary image manipulation
def mask_dilate(mask, radius=2):
    coordinate_profile = np.arange(-np.ceil(radius) + 1, np.ceil(radius))
    x, y = np.meshgrid(coordinate_profile, coordinate_profile)
    d_squared = x**2 + y**2 + 1

    h = np.clip((radius**2 / d_squared - 1) * 1 * np.sqrt(radius), 0, 1)

    return scipy.signal.convolve(mask * 1, h, mode="same") > 0.01


def mask_erode(mask, radius=2):
    coordinate_profile = np.arange(-np.ceil(radius) + 1, np.ceil(radius))
    x, y = np.meshgrid(coordinate_profile, coordinate_profile)
    d_squared = x**2 + y**2 + 1

    h = np.clip((radius**2 / d_squared - 1) * 1 * np.sqrt(radius), 0, 1)

    return scipy.signal.convolve(mask * 1, h, mode="same") > 0.99


def mask_dilate_erode(mask, radius=2):
    return mask_erode(mask_dilate(mask, radius), radius)


# RGB image preparation
def quantile_clip(arr, q_min=0.05, q_max=0.95, q_nan=None, clip_skip=True):
    if clip_skip:
        min_val = np.quantile(arr[(arr != np.min(arr)) & ~np.isnan(arr)], q_min)
        max_val = np.quantile(arr[(arr != np.max(arr)) & ~np.isnan(arr)], q_max)
    else:
        min_val = np.quantile(arr[~np.isnan(arr)], q_min)
        max_val = np.quantile(arr[~np.isnan(arr)], q_max)
    if not q_nan == None:
        nan_val = np.quantile(arr[~np.isnan(arr)], q_nan)
    else:
        nan_val = np.nan

    new_arr = copy.deepcopy(arr)

    new_arr[arr < min_val] = min_val
    new_arr[arr > max_val] = max_val
    new_arr[np.isnan(arr)] = nan_val

    return new_arr


def stack_floats_to_rgb24(
    image_list, q_clip=(0.05, 0.95, 0), scale_factor="auto_individual"
):
    im_out = np.zeros(
        [image_list[0].shape[0], image_list[0].shape[1], 3], dtype=np.uint8
    )
    if scale_factor == "auto_individual":
        if not q_clip == None:
            for chan in range(np.minimum(3, len(image_list))):
                clipped_image_list = [
                    quantile_clip(
                        image,
                        q_min=q_clip[0],
                        q_max=q_clip[1],
                        q_nan=q_clip[-1],
                    )
                    for image in image_list
                ]
                im_out[:, :, chan] = (
                    255
                    * (
                        clipped_image_list[chan]
                        - np.min(clipped_image_list[chan])
                    )
                    / np.max(
                        clipped_image_list[chan]
                        - np.min(clipped_image_list[chan])
                    )
                ).astype(np.uint8)
        else:
            for chan in range(np.minimum(3, len(image_list))):
                im_out[:, :, chan] = (
                    255
                    * (image_list[chan] - np.min(image_list[chan]))
                    / np.max(image_list[chan] - np.min(image_list[chan]))
                ).astype(np.uint8)
    if scale_factor == "auto_same":
        raw_stack = np.dstack(image_list)
        if q_clip == None:
            raw_stack = np.dstack(image_list)
            im_out = (
                255
                * (raw_stack - np.min(raw_stack))
                / np.max((raw_stack - np.min(raw_stack)))
            ).astype(np.uint8)
        else:
            raw_stack = quantile_clip(
                np.dstack(image_list),
                q_min=q_clip[0],
                q_max=q_clip[1],
                q_nan=q_clip[-1],
            )
            im_out = (
                255
                * (raw_stack - np.min(raw_stack))
                / np.max((raw_stack - np.min(raw_stack)))
            ).astype(np.uint8)
    return im_out

def read_image(path):
    # planning to expand this to choose tifffile, or pydicom, etc., depending 
    # on file extension.
    return plt.imread(path)
