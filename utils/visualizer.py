"""
This module defines the Visualizer class, which is used to draw the visualizer on the canvas.
"""
from typing import Dict, Optional
import numpy as np
import cv2


class Visualizer:
    """
    This class is used to draw the visualizer on the canvas.
    Will be replaced with a more general solution in the future to allow for more customization.
    """
    def __init__(self, base_size, max_size, color, dot_count, width, height):
        self.base_size = base_size
        self.max_size = max_size
        self.color = color
        self.dot_count = dot_count
        self.width = width
        self.height = height
        self.cached_dot_positions = None
        self.cached_resized_drawing = {}

    def initialize_static_values(self: "Visualizer") -> None:
        """
        Initializes static values for the visualizer.
        :return: None.
        """
        # Calculate and store dot positions
        x_positions = (self.width / self.dot_count[0]) * np.arange(self.dot_count[0]) + (
                self.width / self.dot_count[0] / 2)
        y_positions = (self.height / self.dot_count[1]) * np.arange(self.dot_count[1]) + (
                self.height / self.dot_count[1] / 2)
        grid_x, grid_y = np.meshgrid(x_positions, y_positions)
        self.cached_dot_positions = [(grid_x[y, x], grid_y[y, x]) for x in range(self.dot_count[0]) for y in
                                     range(self.dot_count[1])]

    def draw_visualizer(self: "Visualizer", canvas: np.ndarray, frequency_data: Dict[float, float],
                        custom_drawing: Optional[np.ndarray] = None) -> None:
        """
        Draws the visualizer on the canvas (a single frame).
        :param canvas: The canvas to draw on.
        :param frequency_data: The frequency data to use for drawing which correlates to the loudness + frequency.
        :param custom_drawing: A custom drawing to use instead of the default circle.
        :return: None.
        """
        # Calculate and store dot positions
        dot_count_x = self.dot_count[0]
        dot_count_y = self.dot_count[1]

        # Precompute log frequencies
        freq_keys = np.array(list(frequency_data.keys()))
        start_freq = freq_keys[freq_keys > 0][0] if freq_keys[freq_keys > 0].size > 0 else 1.0
        end_freq = freq_keys[-1]
        log_freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), dot_count_x)

        # Find the maximum and minimum loudness values, ignoring -80 dB
        freq_bands = np.array([frequency_data[key] for key in freq_keys if key > 0])  # Ignore 0 Hz
        max_loudness = np.max(freq_bands)
        filtered_loudness = freq_bands[freq_bands > -80]
        min_loudness = np.min(filtered_loudness) if filtered_loudness.size > 0 else -80

        # Precompute loudness values
        loudness_values = {}
        for x in range(dot_count_x):
            lower_bound = log_freqs[x]
            upper_bound = log_freqs[x + 1] if x < dot_count_x - 1 else end_freq + 1
            band_freqs = [freq for freq in freq_keys if lower_bound <= freq < upper_bound]
            if not band_freqs:
                closest_freq = min(freq_keys, key=lambda f: abs(f - lower_bound))
                band_freqs = [closest_freq]

            band_loudness = [frequency_data[freq] for freq in band_freqs]
            avg_loudness = np.mean(band_loudness) if band_loudness else -80
            loudness_values[x] = avg_loudness

        cached_dot_sizes = {}
        for i, (pos_x, pos_y) in enumerate(self.cached_dot_positions):
            column = i // dot_count_y  # Ensure the correct column is computed

            if column not in cached_dot_sizes:
                avg_loudness = loudness_values[column]
                # Scale the loudness to the dot size
                scaled_loudness = (avg_loudness - min_loudness) / (
                        max_loudness - min_loudness) if max_loudness != min_loudness else 0
                dot_size = self.base_size + scaled_loudness * (self.max_size - self.base_size)
                dot_size = min(max(dot_size, self.base_size), self.max_size)

                cached_dot_sizes[column] = dot_size
            else:
                dot_size = cached_dot_sizes[column]

            # Convert dot size to integer and calculate the center position
            dot_size = int(dot_size)
            center = (int(pos_x), int(pos_y))
            if custom_drawing is not None:
                if dot_size not in self.cached_resized_drawing:
                    self.cached_resized_drawing[dot_size] = cv2.resize(custom_drawing, (dot_size, dot_size),
                                                                       interpolation=cv2.INTER_LANCZOS4)
                resized_custom_drawing = self.cached_resized_drawing[dot_size]

                center_x, center_y = int(pos_x), int(pos_y)
                half_dot_size = dot_size // 2

                # Calculate bounds on the canvas
                start_x = max(center_x - half_dot_size, 0)
                end_x = min(center_x + half_dot_size, canvas.shape[1])
                start_y = max(center_y - half_dot_size, 0)
                end_y = min(center_y + half_dot_size, canvas.shape[0])

                # Calculate corresponding bounds on the resized image
                img_start_x = max(half_dot_size - (center_x - start_x), 0)
                img_end_x = img_start_x + (end_x - start_x)
                img_start_y = max(half_dot_size - (center_y - start_y), 0)
                img_end_y = img_start_y + (end_y - start_y)

                # Place the image slice onto the canvas
                canvas[start_y:end_y, start_x:end_x] = resized_custom_drawing[img_start_y:img_end_y,
                                                                              img_start_x:img_end_x]
            else:
                cv2.circle(canvas, center, dot_size // 2, self.color, -1)
