"""An example focused on training a network to denoise a time series."""

from typing import Dict

import matplotlib.pyplot as plt
import torch as th
from torch.func import grad_and_value
from tqdm import tqdm
import os


def sigmoid(x: th.Tensor) -> th.Tensor:
    """Define logistic sigmoid following 1 / (1 + e^(-x)).

    Args:
        x (th.Tensor): Input Tensor.

    Returns:
        th.Tensor: Sigmoid activated input.
    """
    # TODO: 1. Implement sigmoid activation function.
    return 0.0


def net(params: Dict, x: th.Tensor) -> th.Tensor:
    """Set up a single layer network.

    Args:
        params (Dict): Dictionary containing W1, b, and W2.
        x (th.Tensor): Network input.

    Returns:
        th.Tensor: Network prediction.
    """
    # TODO: 2. Implement a single layer pass.
    return None


def cost(y: th.Tensor, h: th.Tensor) -> th.Tensor:
    """Compute Squared Error loss.

    Args:
        y (th.Tensor): Ground truth output.
        h (th.Tensor): Network predicted output.

    Returns:
        th.Tensor: Squared Error.
    """
    # TODO: 4. Implement Squared Error loss.
    return 0.0


def net_cost(params: Dict, x: th.Tensor, y: th.Tensor) -> th.Tensor:
    """Evaluate the network and compute the loss.

    Args:
        params (Dict): Dictionary containing W1, b, and W2.
        x (th.Tensor): Network input.
        y (th.Tensor): Desired output.

    Returns:
        th.Tensor: Squared Error.
    """
    # TODO: 5. Call network, compute and return the loss.
    return None


if __name__ == "__main__":
    # TODO: Use th.manual_seed as 42 to set the seed for the network initialization
    pass
    # TODO: Choose a suitable step size (the step size is typically a small positive value,
    # you can try values between 1e-1 to 1e-5)
    step_size = 0.0
    iterations = 100
    input_neurons = output_neurons = 200
    # TODO: Choose a proper network size.
    hidden_neurons = 0

    x = th.linspace(-3 * th.pi, 3 * th.pi, 200)
    y = th.cos(x)

    # TODO: 3. Initialize the parameters
    W1 = None
    b = None
    W2 = None

    # TODO: Instantiate grad_and_value function. The grad_and_value function takes a function as input and returns another function
    # that computes both the gradients and the value of the input function. Thus we can use it to compute the gradients of the cost function with respect to the network parameters.
    value_grad = None

    # Training loop
    for i in (pbar := tqdm(range(iterations))):
        # Set a new seed each loop to generate different noise
        th.manual_seed(i)
        y_noise = y + th.randn([200])

        # 6.
        # TODO: Compute loss and gradients

        # TODO: Update parameters using SGD

    # TODO: 7. Compute test y_hat using y_noise and converged parameters
    y_hat = None

    plt.title("Denoising a cosine")
    plt.plot(x, y, label="solution")
    plt.plot(x, y_hat, "x", label="fit")
    plt.plot(x, y_noise, label="input")
    plt.legend()
    plt.grid()
    os.makedirs("./figures", exist_ok=True)
    plt.savefig("./figures/Denoise.png", dpi=600, bbox_inches="tight")
    plt.show()
    print("Done")
