import time
from typing import Optional


def print_progress_bar(current_iteration: int, total_iterations: int, bar_length: int = 50,
                       start_time: Optional[float] = None, end: str = ''):
    progress_percentage = (current_iteration / total_iterations) * 100
    completed_length = int(bar_length * current_iteration // total_iterations)
    progress_bar = '█' * completed_length + '░' * (bar_length - completed_length)

    elapsed_time = None
    estimated_remaining_time = None
    iterations_per_sec = None
    if start_time is not None:
        elapsed_time = time.time() - start_time
        if current_iteration > 0:
            estimated_total_time = elapsed_time / current_iteration * total_iterations
            estimated_remaining_time = estimated_total_time - elapsed_time
            iterations_per_sec = current_iteration / elapsed_time
        else:
            estimated_remaining_time = None

    time_string = ''
    if estimated_remaining_time is not None and iterations_per_sec is not None:
        time_string = f'[{elapsed_time:.2f}s/{estimated_remaining_time:.2f}s, {iterations_per_sec:.2f}it/s]'
    print(f'\r{progress_percentage:3.0f}%|{progress_bar}| {current_iteration}/{total_iterations} {time_string}', end=end, flush=True)
