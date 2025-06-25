from Layers.Base import Baselayer
import numpy as np

class Flatten(Baselayer):
    def __init__(self):
        super().__init__()  # Initializes 'trainable = False' from BaseLayer
        self._original_shape = None

    def forward(self, input_tensor):
        self._original_shape = input_tensor.shape
        batch_size = self._original_shape[0]
        return input_tensor.reshape(batch_size, -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self._original_shape)