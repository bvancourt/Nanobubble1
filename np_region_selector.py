"""
GUI for selecting regions of interest on numpy images.
"""

import tkinter as tk
import warnings
import numpy as np
import pixelhelp as ph
import ctypes as ct
from copy import deepcopy
import importlib

importlib.reload(ph)

class NpRegionSelector:
    def __init__(
        self,
        source_image,
        selection_mode="rectangle",
        initial_height=512,
        initial_width=512,
        view_mode="autozoom",
        pixel_aspect_ratio=1,
        return_points=False,
    ):
        self.return_points = return_points
        if source_image.dtype == np.uint8:
            self.source_image = source_image
        elif source_image.dtype == np.uint16:
            self.source_image = (source_image // 2**8).astype(np.uint8)
        elif source_image.dtype in [np.float16, np.float32, np.float16]:
            if (np.max(source_image[~np.isnan(source_image)]) <= 1) and (
                np.min(source_image[~np.isnan(image)]) >= 0
            ):
                # In this case, assume that 0 is should be black and one should
                #   be white.
                self.source_image = (image * 255).astype(np.uint8)
            else:
                self.source_image = (
                    255
                    * (source_image - np.nanmin(source_image))
                    / (np.nanmax(source_image) - np.nanmin(source_image))
                ).astype(np.uint8)

        else:
            print("Image data type {} not supported by NpRegionSelector")

        if len(self.source_image.shape) == 2:
            self.source_image = np.dstack(
                [self.source_image, self.source_image, self.source_image]
            )

        self.window_height = initial_height
        self.window_width = initial_width
        self.selection_mode = selection_mode
        self.view_mode = view_mode
        self.selection_mode = selection_mode
        self.pixel_aspect_ratio = pixel_aspect_ratio

        if self.selection_mode == "rectangle":
            self.source_space_first_corner = (0, 0)
            self.source_space_second_corner = (0, 0)
        elif self.selection_mode == "triangle_xor":
            self.source_j, self.source_i = np.meshgrid(
                np.arange(self.source_image.shape[1]),
                np.arange(self.source_image.shape[0]),
            )
            self.dot_radius = 4
            self.clicked_points = []
            self.dots = []
        self.lines = []

    def run(self):
        def resize(event):
            nonlocal self
            if (event.widget == self.canvas) & (
                (event.height != self.window_height)
                | (event.width != self.window_width)
            ):
                self.window_height, self.window_width = (
                    event.height,
                    event.width,
                )
                if self.view_mode == "autozoom":
                    x, y = ph.autozoom_grid(
                        self.source_image,
                        [self.window_height, self.window_width],
                        pixel_aspect_ratio=self.pixel_aspect_ratio,
                    )
                    source_copy = deepcopy(self.source_image)
                    source_copy[~self.region_mask] = (
                        source_copy[~self.region_mask] // 2
                    )
                    self.img = ph.np_image_to_tk(
                        ph.im_resamp(source_copy, x, y)
                    )
                    self.canvas.itemconfig(self.im_id, image=self.img)

                    for line in self.lines:
                        self.canvas.delete(line)
                    self.lines = []

                    if self.selection_mode == "rectangle":
                        x0, y0 = ph.source_to_autozoom_coords(
                            self.source_space_first_corner[0],
                            self.source_space_first_corner[1],
                            self.source_image,
                            [self.window_height, self.window_width],
                            method="pad",
                            pixel_aspect_ratio=self.pixel_aspect_ratio,
                        )
                        x1, y1 = ph.source_to_autozoom_coords(
                            self.source_space_second_corner[0],
                            self.source_space_second_corner[1],
                            self.source_image,
                            [self.window_height, self.window_width],
                            method="pad",
                            pixel_aspect_ratio=self.pixel_aspect_ratio,
                        )
                        self.lines.append(
                            self.canvas.create_line(
                                y0, x0, y0, x1, fill="orange", width=2
                            )
                        )
                        self.lines.append(
                            self.canvas.create_line(
                                y0, x0, y1, x0, fill="orange", width=2
                            )
                        )
                        self.lines.append(
                            self.canvas.create_line(
                                y1, x0, y1, x1, fill="orange", width=2
                            )
                        )
                        self.lines.append(
                            self.canvas.create_line(
                                y0, x1, y1, x1, fill="orange", width=2
                            )
                        )

                    elif self.selection_mode == "triangle_xor":

                        for dot in self.dots:
                            self.canvas.delete(dot)
                        self.dots = []
                        for i, source_point in enumerate(self.clicked_points):
                            screen_x, screen_y = ph.source_to_autozoom_coords(
                                source_point[0],
                                source_point[1],
                                self.source_image,
                                [self.window_height, self.window_width],
                                method="pad",
                                pixel_aspect_ratio=self.pixel_aspect_ratio,
                            )

                            if i == 0:
                                line_color = "green"
                                dot_color = "blue"
                            else:
                                line_color = "orange"
                                dot_color = "orange"

                            self.dots.append(
                                self.canvas.create_oval(
                                    screen_y - self.dot_radius,
                                    screen_x - self.dot_radius,
                                    screen_y + self.dot_radius,
                                    screen_x + self.dot_radius,
                                    fill=dot_color,
                                )
                            )
                            if len(self.clicked_points) > 1:
                                last_screen_x, last_screen_y = (
                                    ph.source_to_autozoom_coords(
                                        self.clicked_points[i - 1][0],
                                        self.clicked_points[i - 1][1],
                                        self.source_image,
                                        [self.window_height, self.window_width],
                                        method="pad",
                                        pixel_aspect_ratio=(
                                            self.pixel_aspect_ratio
                                        ),
                                    )
                                )

                            self.lines.append(
                                self.canvas.create_line(
                                    last_screen_y,
                                    last_screen_x,
                                    screen_y,
                                    screen_x,
                                    fill=line_color,
                                    width=2,
                                )
                            )

                else:
                    warnings.warn(
                        'NpRegionSelector unable to resize becuase veiw_mode '
                        + f'"{self.view_mode}" is unrecognized.'
                    )

        def left_click(event):
            nonlocal self
            if event.widget == self.canvas:
                xim, yim = ph.autozoom_to_source_coords(
                    event.y,
                    event.x,
                    self.source_image,
                    [self.window_height, self.window_width],
                    pixel_aspect_ratio=self.pixel_aspect_ratio,
                )

                if self.selection_mode == "rectangle":
                    self.source_space_first_corner = (int(xim), int(yim))
                elif self.selection_mode == "triangle_xor":
                    self.clicked_points.append((int(xim), int(yim)))

                    if len(self.clicked_points) == 3:
                        self.region_mask = ~self.region_mask
                    if len(self.clicked_points) >= 3:
                        self.region_mask ^= ph.point_in_triangle(
                            self.source_i,
                            self.source_j,
                            [
                                self.clicked_points[-2],
                                self.clicked_points[-1],
                                self.clicked_points[0],
                            ],
                            count_exact_edge="triangle_xor",
                        )
                        source_copy = deepcopy(self.source_image)
                        source_copy[~self.region_mask] = (
                            source_copy[~self.region_mask] // 2
                        )
                        x, y = ph.autozoom_grid(
                            self.source_image,
                            [self.window_height, self.window_width],
                            pixel_aspect_ratio=self.pixel_aspect_ratio,
                        )
                        self.img = ph.np_image_to_tk(
                            ph.im_resamp(source_copy, x, y)
                        )
                        self.canvas.itemconfig(self.im_id, image=self.img)

                    for dot in self.dots:
                        self.canvas.delete(dot)
                    self.dots = []
                    for line in self.lines:
                        self.canvas.delete(line)
                    self.lines = []
                    for i, source_point in enumerate(self.clicked_points):
                        screen_x, screen_y = ph.source_to_autozoom_coords(
                            source_point[0],
                            source_point[1],
                            self.source_image,
                            [self.window_height, self.window_width],
                            method="pad",
                            pixel_aspect_ratio=self.pixel_aspect_ratio,
                        )

                        if i == 0:
                            line_color = "green"
                            dot_color = "blue"
                        else:
                            line_color = "orange"
                            dot_color = "orange"

                        self.dots.append(
                            self.canvas.create_oval(
                                screen_y - self.dot_radius,
                                screen_x - self.dot_radius,
                                screen_y + self.dot_radius,
                                screen_x + self.dot_radius,
                                fill=dot_color,
                            )
                        )
                        if len(self.clicked_points) > 1:
                            last_screen_x, last_screen_y = (
                                ph.source_to_autozoom_coords(
                                    self.clicked_points[i - 1][0],
                                    self.clicked_points[i - 1][1],
                                    self.source_image,
                                    [self.window_height, self.window_width],
                                    method="pad",
                                    pixel_aspect_ratio=self.pixel_aspect_ratio,
                                )
                            )

                            self.lines.append(
                                self.canvas.create_line(
                                    last_screen_y,
                                    last_screen_x,
                                    screen_y,
                                    screen_x,
                                    fill=line_color,
                                    width=2,
                                )
                            )

        def left_release(event):
            nonlocal self
            if event.widget == self.canvas:
                xim, yim = ph.autozoom_to_source_coords(
                    event.y,
                    event.x,
                    self.source_image,
                    [self.window_height, self.window_width],
                    pixel_aspect_ratio=self.pixel_aspect_ratio,
                )

                if self.selection_mode == "rectangle":
                    self.source_space_second_corner = (int(xim), int(yim))
                    self.region_mask = np.zeros(
                        self.source_image.shape[:2], dtype=np.bool_
                    )

                    ROI_x0 = np.min(
                        [
                            self.source_space_first_corner[0],
                            self.source_space_second_corner[0],
                        ]
                    )
                    ROI_x1 = np.max(
                        [
                            self.source_space_first_corner[0],
                            self.source_space_second_corner[0],
                        ]
                    )
                    ROI_y0 = np.min(
                        [
                            self.source_space_first_corner[1],
                            self.source_space_second_corner[1],
                        ]
                    )
                    ROI_y1 = np.max(
                        [
                            self.source_space_first_corner[1],
                            self.source_space_second_corner[1],
                        ]
                    )
                    self.region_mask[ROI_x0:ROI_x1, ROI_y0:ROI_y1] = True

                    x, y = ph.autozoom_grid(
                        self.source_image,
                        [self.window_height, self.window_width],
                        pixel_aspect_ratio=self.pixel_aspect_ratio,
                    )
                    source_copy = deepcopy(self.source_image)
                    source_copy[~self.region_mask] = (
                        source_copy[~self.region_mask] // 2
                    )
                    self.img = ph.np_image_to_tk(
                        ph.im_resamp(source_copy, x, y)
                    )
                    self.canvas.itemconfig(self.im_id, image=self.img)

                    for line in self.lines:
                        self.canvas.delete(line)
                    self.lines = []

                    x0, y0 = ph.source_to_autozoom_coords(
                        self.source_space_first_corner[0],
                        self.source_space_first_corner[1],
                        self.source_image,
                        [self.window_height, self.window_width],
                        method="pad",
                        pixel_aspect_ratio=self.pixel_aspect_ratio,
                    )
                    x1, y1 = ph.source_to_autozoom_coords(
                        self.source_space_second_corner[0],
                        self.source_space_second_corner[1],
                        self.source_image,
                        [self.window_height, self.window_width],
                        method="pad",
                        pixel_aspect_ratio=self.pixel_aspect_ratio,
                    )
                    self.lines.append(
                        self.canvas.create_line(
                            y0, x0, y0, x1, fill="orange", width=2
                        )
                    )
                    self.lines.append(
                        self.canvas.create_line(
                            y0, x0, y1, x0, fill="orange", width=2
                        )
                    )
                    self.lines.append(
                        self.canvas.create_line(
                            y1, x0, y1, x1, fill="orange", width=2
                        )
                    )
                    self.lines.append(
                        self.canvas.create_line(
                            y0, x1, y1, x1, fill="orange", width=2
                        )
                    )

                elif self.selection_mode == "triangle_xor":
                    pass

        def left_drag(event):
            nonlocal self
            if event.widget == self.canvas:
                for line in self.lines:
                    self.canvas.delete(line)
                self.lines = []

                if self.selection_mode == "rectangle":
                    x0, y0 = ph.source_to_autozoom_coords(
                        self.source_space_first_corner[0],
                        self.source_space_first_corner[1],
                        self.source_image,
                        [self.window_height, self.window_width],
                        method="pad",
                        pixel_aspect_ratio=self.pixel_aspect_ratio,
                    )
                    self.lines.append(
                        self.canvas.create_line(
                            y0, x0, y0, event.y, fill="green", width=2.5
                        )
                    )
                    self.lines.append(
                        self.canvas.create_line(
                            y0, x0, event.x, x0, fill="green", width=2.5
                        )
                    )
                    self.lines.append(
                        self.canvas.create_line(
                            event.x,
                            x0,
                            event.x,
                            event.y,
                            fill="green",
                            width=2.5,
                        )
                    )
                    self.lines.append(
                        self.canvas.create_line(
                            y0,
                            event.y,
                            event.x,
                            event.y,
                            fill="green",
                            width=2.5,
                        )
                    )

                elif self.selection_mode == "triangle_xor":
                    pass

        self.root = tk.Tk()
        self.root.title("ROI Selector")

        background_color = "#777777"  # this is only visible in glitchy-looking 
        #   situations, so better if it doesn't stand out...
        self.root.config(bg=background_color)

        self.canvas = tk.Canvas(
            self.root, height=self.window_height, width=self.window_width
        )
        self.canvas.bind("<Configure>", resize)
        self.canvas.bind("<Button-1>", left_click)
        if not self.selection_mode == "triangle_xor":
            self.canvas.bind("<ButtonRelease-1>", left_release)
            self.canvas.bind("<B1-Motion>", left_drag)
        self.canvas.place(relwidth=1, relheight=1)
        x, y = np.meshgrid(
            np.arange(self.window_width), np.arange(self.window_height)
        )
        self.img = ph.np_image_to_tk(self.source_image)
        self.im_id = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

        self.region_mask = np.ones(self.source_image.shape[:2], dtype=np.bool_)

        self.root.mainloop()

        if self.return_points and (self.selection_mode == "triangle_xor"):
            return self.region_mask, self.clicked_points
        else:
            return self.region_mask


if __name__ == "__main__":
    # test/usage example
    source_image = (ph.read_image("BigTestImage.jpg") * 2**8).astype(np.uint16)
    ROI_selector = NpRegionSelector(
        source_image, pixel_aspect_ratio=1.56789, selection_mode="triangle_xor"
    )
    ROI_mask = ROI_selector.run()
