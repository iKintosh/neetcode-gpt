import numpy as np
from numpy.typing import NDArray

def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))

def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(x, 0)


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        z = x @ w + b
        if activation == "relu":
            return np.round(relu(z), 5)
        elif activation == "sigmoid":
            return np.round(sigmoid(z), 5)
