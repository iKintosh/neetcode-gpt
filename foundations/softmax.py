import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_z = max(z)
        norm_z = z - max_z
        softmax_values = np.exp(norm_z) / np.sum(np.exp(norm_z))
        return np.round(softmax_values, 4)
