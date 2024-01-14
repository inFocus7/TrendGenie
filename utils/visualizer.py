import numpy as np
import cv2


class Visualizer:
    def __init__(self, base_size, max_size, color, dot_count, width, height):
        self.base_size = base_size
        self.max_size = max_size
        self.color = color # This is CV2, so will need to be BGR
        self.dot_count = dot_count
        self.width = width
        self.height = height
        self.cached_dot_positions = None

    def initialize_static_values(self):
        # Calculate and store dot positions
        x_positions = (self.width / self.dot_count[0]) * np.arange(self.dot_count[0]) + (
                    self.width / self.dot_count[0] / 2)
        y_positions = (self.height / self.dot_count[1]) * np.arange(self.dot_count[1]) + (
                    self.height / self.dot_count[1] / 2)
        grid_x, grid_y = np.meshgrid(x_positions, y_positions)
        self.cached_dot_positions = [(grid_x[y, x], grid_y[y, x]) for x in range(self.dot_count[0]) for y in
                                     range(self.dot_count[1])]

    def draw_visualizer(self, canvas, frequency_data, custom_drawing=None):
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
                pass
                # custom_drawing = custom_drawing.resize((int(dot_size), int(dot_size)), Image.LANCZOS)
                # canvas.paste(custom_drawing, (int(pos_x - dot_size / 2), int(pos_y - dot_size / 2)),
                #                    custom_drawing)
            else:
                cv2.circle(canvas, center, dot_size // 2, self.color, -1)
