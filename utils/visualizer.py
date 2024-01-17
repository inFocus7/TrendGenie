"""
This module defines the Visualizer class, which is used to draw the visualizer on the canvas.
"""
from typing import Dict, Optional
import numpy as np
import cv2
from utils import dataclasses, image as image_utils


class Visualizer:
    """
    This class is used to draw the visualizer on the canvas.
    Will be replaced with a more general solution in the future to allow for more customization.
    """

    def __init__(self, dot_size: dataclasses.MinMax, color, dot_count: dataclasses.RowCol, size: dataclasses.Size):
        self.dot_size = dot_size
        self.color = color
        self.dot_count = dot_count
        self.size = size
        self.cached_dot_positions = None
        self.cached_resized_drawing = {}

    def initialize_static_values(self: "Visualizer") -> None:
        """
        Initializes static values for the visualizer.
        :return: None.
        """
        # Calculate and store dot positions
        x_positions = (self.size.width / self.dot_count.col) * np.arange(self.dot_count.col) + (
                self.size.width / self.dot_count.col / 2)
        y_positions = (self.size.height / self.dot_count.row) * np.arange(self.dot_count.row) + (
                self.size.height / self.dot_count.row / 2)
        grid_x, grid_y = np.meshgrid(x_positions, y_positions)
        self.cached_dot_positions = [(grid_x[y, x], grid_y[y, x]) for x in range(self.dot_count.col) for y in
                                     range(self.dot_count.row)]

    def _get_loudness(self, frequency_data: Dict[float, float]) -> (dataclasses.MinMax, Dict[int, int]):
        """
        Calculates the loudness values for each column.
        :param frequency_data: The frequency data to use for drawing which correlates to the loudness + frequency.
        :return: A tuple containing the loudness min/max and the loudness values for each column.
        """
        # Precompute log frequencies
        freq_keys = np.array(list(frequency_data.keys()))
        start_freq = freq_keys[freq_keys > 0][0] if freq_keys[freq_keys > 0].size > 0 else 1.0
        end_freq = freq_keys[-1]
        log_freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), self.dot_count.col)

        # Find the maximum and minimum loudness values, ignoring -80 dB
        freq_bands = np.array([frequency_data[key] for key in freq_keys if key > 0])  # Ignore 0 Hz
        filtered_loudness = freq_bands[freq_bands > -80]
        loudness_min_max = dataclasses.MinMax(np.min(filtered_loudness) if filtered_loudness.size > 0 else -80,
                                              np.max(freq_bands))

        # Precompute loudness values
        loudness_values = {}
        for x in range(self.dot_count.col):
            bounds = {
                "lower": log_freqs[x],
                "upper": log_freqs[x + 1] if x < self.dot_count.col - 1 else end_freq + 1
            }
            band_freqs = [freq for freq in freq_keys if bounds.get("lower") <= freq < bounds.get("upper")]
            if not band_freqs:
                closest_freq = min(freq_keys, key=lambda f, lb=bounds.get("lower"): abs(f - lb))
                band_freqs = [closest_freq]

            band_loudness = [frequency_data[freq] for freq in band_freqs]
            avg_loudness = np.mean(band_loudness) if band_loudness else -80
            loudness_values[x] = avg_loudness

        return loudness_min_max, loudness_values

    def _calculate_dot_size(self: "Visualizer", column: int, loudness: dataclasses.MinMax,
                            loudness_values: Dict[int, int]) -> int:
        """
        Calculates the dot size for a given column.
        :param loudness: The loudness min/max.
        :param loudness_values: The loudness values for each column.
        :return: The dot size.
        """
        # Scale the loudness to the dot size
        scaled_loudness = (loudness_values[column] - loudness.min) / (
                loudness.max - loudness.min) if loudness.max != loudness.min else 0
        dot_size = self.dot_size.min + scaled_loudness * (self.dot_size.max - self.dot_size.min)
        return min(max(dot_size, self.dot_size.min), self.dot_size.max)

    def _draw_custom_drawing(self: "Visualizer", canvas: np.ndarray, start_pos: dataclasses.Position,
                             end_pos: dataclasses.Position, img_start_pos: dataclasses.Position,
                             img_end_pos: dataclasses.Position, dot_size: int,
                             custom_drawing_overlap: bool) -> np.ndarray:
        """
        Draws the custom drawing on the canvas.
        :param canvas: The canvas to draw on.
        :param start_pos: The start position on the canvas.
        :param end_pos: The end position on the canvas.
        :param img_start_pos: The start position on the resized image.
        :param img_end_pos: The end position on the resized image.
        :param dot_size: The dot size.
        :param custom_drawing_overlap: Whether overlapped custom drawings should alpha blend.
        :return: The canvas with the custom drawing drawn on it.
        """
        drawing_slice = self.cached_resized_drawing[dot_size][img_start_pos.y:img_end_pos.y,
                        img_start_pos.x:img_end_pos.x]

        if custom_drawing_overlap:
            canvas_slice = canvas[start_pos.y:end_pos.y, start_pos.x:end_pos.x]
            return image_utils.blend_alphas(canvas_slice, drawing_slice)

        return drawing_slice

    def draw_visualizer(self: "Visualizer", canvas: np.ndarray, frequency_data: Dict[float, float],
                        custom_drawing: Optional[np.ndarray] = None, custom_drawing_overlap: bool = False) -> None:
        """
        Draws the visualizer on the canvas (a single frame).
        :param custom_drawing_overlap: Whether to overlap the custom drawing should alpha blend when overlapping.
        :param canvas: The canvas to draw on.
        :param frequency_data: The frequency data to use for drawing which correlates to the loudness + frequency.
        :param custom_drawing: A custom drawing to use instead of the default circle.
        :return: None.
        """
        loudness, loudness_values = self._get_loudness(frequency_data)

        cached_dot_sizes = {}
        for i, (pos_x, pos_y) in enumerate(self.cached_dot_positions):
            column = i // self.dot_count.row  # Ensure the correct column is computed

            if column not in cached_dot_sizes:
                cached_dot_sizes[column] = self._calculate_dot_size(column, loudness, loudness_values)

            dot_size = int(cached_dot_sizes[column])
            center_pos = dataclasses.Position(int(pos_x), int(pos_y))
            if custom_drawing is not None:
                if dot_size not in self.cached_resized_drawing:
                    if dot_size == 0:
                        self.cached_resized_drawing[dot_size] = np.zeros((1, 1, 4), dtype=np.uint8)
                    else:
                        self.cached_resized_drawing[dot_size] = cv2.resize(custom_drawing, (dot_size, dot_size),
                                                                           interpolation=cv2.INTER_LANCZOS4)

                half_dot_size = dot_size // 2
                # Calculate bounds on the canvas
                start_pos = dataclasses.Position(max(center_pos.x - half_dot_size, 0),
                                                 max(center_pos.y - half_dot_size, 0))
                end_pos = dataclasses.Position(min(center_pos.x + half_dot_size, canvas.shape[1]), min(
                    center_pos.y + half_dot_size, canvas.shape[0]))

                # Calculate corresponding bounds on the resized image
                img_start_pos = dataclasses.Position(max(half_dot_size - (center_pos.x - start_pos.x), 0),
                                                     max(half_dot_size - (center_pos.y - start_pos.y), 0))
                img_end_pos = dataclasses.Position(img_start_pos.x + (end_pos.x - start_pos.x),
                                                   img_start_pos.y + (end_pos.y - start_pos.y))

                canvas[start_pos.y:end_pos.y, start_pos.x:end_pos.x] = self._draw_custom_drawing(canvas, start_pos,
                                                                                                 end_pos, img_start_pos,
                                                                                                 img_end_pos, dot_size,
                                                                                                 custom_drawing_overlap)
            else:
                cv2.circle(canvas, (center_pos.x, center_pos.y), dot_size // 2, self.color, -1)
