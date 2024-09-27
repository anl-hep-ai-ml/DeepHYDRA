
import logging
from collections.abc import Callable
from collections import deque

import numpy as np
import pandas as pd

class ReducedDataBuffer():
    def __init__(self,
                 size: int,
                 buffer_filled_callback: Callable = None) -> None:

        self._size = size
        self._buffer_filled_callback = buffer_filled_callback

        self._buffer = None  # Will be initialized when the first data arrives
        self._indices = [None] * self._size  # List to store indices (timestamps)
        self._current_index = 0  # Keep track of the current index in the buffer

        self._logger = logging.getLogger(__name__)
        self._buffer_filled_feedback_given = False

    def push(self, data: pd.DataFrame):
        data_size = len(data)

        if self._buffer is None:
            # Initialize the buffer DataFrame with the appropriate columns and size
            num_columns = data.shape[1]
            column_names = data.columns
            # Create an empty DataFrame with preallocated space
            self._buffer = pd.DataFrame(
                np.empty((self._size, num_columns), dtype=np.float32),
                columns=column_names
            )
            self._indices = [None] * self._size  # Reset indices list

        if data_size > 1:
            # For data with multiple rows
            for idx, row in data.iterrows():
                if self._current_index < self._size:
                    self._buffer.iloc[self._current_index] = row.values
                    self._indices[self._current_index] = idx  # Store the index (timestamp)
                    self._current_index += 1
                else:
                    break  # Buffer is full
        elif data_size == 1:
            # For data with a single row
            self._buffer.iloc[self._current_index] = data.values[0]
            self._indices[self._current_index] = data.index[0]  # Store the index (timestamp)
            self._current_index += 1
        else:
            return  # No data to push

        if self._current_index >= self._size:
            # Buffer is full, set the correct index
            self._buffer.index = pd.Index(self._indices)
            if self._buffer_filled_callback is not None:
                # Pass a copy to avoid unintended modifications
                self._buffer_filled_callback(self._buffer.copy())
            # Reset the buffer index and indices
            self._current_index = 0
            self._indices = [None] * self._size

            if not self._buffer_filled_feedback_given:
                self._logger.info('Buffer filled')
                self._buffer_filled_feedback_given = True
        

    def set_buffer_filled_callback(self, callback: Callable) -> None:
        self._buffer_filled_callback = callback


# class ReducedDataBuffer():
#     def __init__(self,
#                     size: int,
#                     buffer_filled_callback: Callable = None) -> None:

#         self._size = size
#         self._buffer_filled_callback =\
#                     buffer_filled_callback

#         self._buffer = deque([], maxlen=self._size)

#         self._logger = logging.getLogger(__name__)
#         self._buffer_filled_feedback_given = False


#     def push(self, data: pd.DataFrame):
        
#         data_size = len(data)

#         if data_size > 1:
#             for data_row in np.vsplit(data, data_size):
#                 self._buffer.append(data_row)

#         elif data_size == 1:
#             self._buffer.append(data)
#         else:
#             return

#         # print(self._buffer)

#         if len(self._buffer) == self._size:

#             buffer_list = list(self._buffer)

#             buffer_pd = pd.concat((buffer_list))

#             if not isinstance(self._buffer_filled_callback, type(None)):
#                 return self._buffer_filled_callback(buffer_pd)

#             if not self._buffer_filled_feedback_given:
#                 self._logger.info('Buffer filled')
#                 self._buffer_filled_feedback_given = True
    

#     # def push_batch(self, batch_data):
#     #     self._buffer.extend(batch_data)
#     #     if len(self._buffer) >= self._size:
#     #         # Convert buffer to batch and call detect
#     #         batch = self._buffer[:self._size]
#     #         self._buffer = self._buffer[self._size:]
#     #         self.callback(batch)


#     def set_buffer_filled_callback(self,
#                                     callback: Callable) -> None:
#         self._buffer_filled_callback = callback