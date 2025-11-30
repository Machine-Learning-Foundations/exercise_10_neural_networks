## Exercise Neural Networks

The goal of this exercise is to implement a multilayer dense neural network using `torch`.
Type,

```bash
pip install -r requirements.txt
```

into the terminal to install the required software.

Torch will take care of our autograd needs, which means we don't have to manually compute gradients. The documentation is available at https://pytorch.org/docs/stable/index.html. torch.nn, which is a submodule of PyTorch, provides all the necessary modules for implementing neural networks. https://pytorch.org/docs/stable/nn.html hosts the documentation.

### Task 1: Denoising a cosine

To get a notion of how function learning of a dense layer network works on given data, we will first have a look at the example from the lecture. In the following task you will implement gradient descent learning of a dense neural network using `torch` and use it to learn a function, e.g. a cosine.

Open `src/denoise_cosine.py` and go to the `__main__` function. Look at the code that is already there. You can see that a cosine function with a signal length of $n = 200$ samples has already been created in torch. In the for loop, which will be our train loop, some noise is added to the cosine function with `torch.randn`. This will be the noisy signal that the model is supposed to learn the underlying cosine from.

1. Recall the definition of the sigmoid function $\sigma$

   ```math
   \sigma(x) = \frac{1}{1 + e^{-x}}
   ```


   Implement the `sigmoid` function in `src/denoise_cosine.py`.


2. Implement a dense layer in the `net` function of `src/denoise_cosine.py`. The function should return
   ```math
   \mathbf{o} = \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b})
   ```
   where $\mathbf{W}_1\in \mathbb{R}^{m,n}, \mathbf{x}\in\mathbb{R}^n, \mathbf{b}\in\mathbb{R}^m$ and $m$ denotes the number of neurons and $n$ the input signal length. Suppose that the input parameters are stored in a [python dictonary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) with the keys `W_1`, `W_2` and `b`.   Use numpys `@` notation for the matrix product.

3. Use `torch.normal` to initialize your weights. This function will sample the values from a normal distribution. To ensure that the weights are not initialized too high, choose a mean of 0 and a standard deviation of 0.5. For a signal length of $200$ the $W_2$ matrix should have the shape [200, `hidden_neurons`] and $W_1$ a shape of [`hidden_neurons`, 200].

4. Implement and test a squared error cost

   ```math
   C_{\text{se}} = \frac{1}{2} \sum_{k=1}^{n} (\mathbf{y}_k - \mathbf{o}_k)^2
   ```

   The `**` operator serves as the power operator for exponentiation in Python, `torch.sum` allows you to sum up all terms of a tensor.

5. Define the forward pass in the `net_cost` function. The forward pass evaluates the network and the cost function.

6. Train your network to denoise a cosine. To do so, implement gradient descent on the noisy input signal and use `torch.grad_and_value` to compute the gradients and cost at the same time. Remember the gradient descent update rule  

   ```math
   \mathbf{W}_{\tau + 1} = \mathbf{W}_\tau - \epsilon \cdot \delta\mathbf{W}_{\tau}.
   ```


   In the equation above $\mathbf{W} \in \mathbb{R}$ holds for weight matrices and biases $\epsilon$ denotes the step size and $\delta$ the gradient operation with respect to the following weight.  Use the loop to repeat weight updates for multiple operations. Try to train for one hundred updates.

7. At last, compute the network output `y_hat` on the final values to see if the network learned the underlying cosine function. Use `matplotlib.pyplot.plot` to plot the noisy signal and the network output $\mathbf{o}$.

8. Test your code with `nox -r -s test` and run the script with `python ./src/denoise_cosine.py` or by pressing `Ctrl + F5` in Vscode. 



### Task 2: MNIST
In this task we will go one step further. Instead of a cosine function, our neural network will learn how to identify handwritten digits from the [MNSIT dataset](http://yann.lecun.com/exdb/mnist/). For that, we will be using the [torch.nn](https://pytorch.org/docs/stable/nn.html) module. To get started familiarize yourself with the torch.nn to train a fully connected network in `src/mnist.py`. In this script, some functions are already implemented and can be reused. [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) is an elegant way to deal with data batches (Torch takes care of this for us). This task aims to compute gradients and update steps for all batches in the list. If you are coding on bender the function `matplotlib.pyplot.show` doesn't work if you are not connected to the X server of bender. Use e.g. `plt.savefig` to save the figure and view it in vscode.

1. Implement the `normalize_batch` function to ensure approximate standard-normal inputs. Make use of handy torch inbuilt methods (like `torch.mean` and `torch.std`). Normalization requires subtraction of the mean and division by the standard deviation with $i = 1, \dots w$ and $j = 1, \dots h$ with $w$ the image width and $h$ the image height and $k$ running through the batch dimension:

```math
\tilde{{x}}_{ijk} = \frac{x_{ijk} - \mu}{\sigma}
```

2. The forward step requires the `Net` object from its [class](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#define-the-class). It is your fully connected neural network model. Implement a dense network in `Net` of your choosing using a combination of `torch.nn.Linear` and `th.nn.ReLU` or `th.nn.Sigmoid`

3. In `Net` class additionally, implement the `forward` function to compute the network forward pass.

4. Write a `cross_entropy` cost function with $n_o$ the number of labels and $n_b$ in the batched case using
      
   ```math
   C_{\text{ce}}(\mathbf{y},\mathbf{o})=-\frac{1}{n_b}\sum_{i=1}^{n_b}\sum_{k=1}^{n_o}[(\mathbf{y}_{i,k}\ln\mathbf{o}_{i,k})+(\mathbf{1}-\mathbf{y}_{i,k})\ln(\mathbf{1}-\mathbf{o}_{i,k})].
   ```

   If you have chosen to work with ten output neurons. Use `torch.nn.functional.one_hot` to encode the labels.

5. Next we want to be able to do an optimization step with stochastic gradient descent (sgd). Implement `sgd_step`. One way to do this is to iterate over `model.parameters()` and update each parameter individually with its gradient. One can access the gradient for each parameter with `<param>.grad`.

6. To evaluate the network we calculate the accuracy of the network output. Implement `get_acc` to calculate the accuracy given a dataloader containing batches of images and corresponding labels. More about dataloaders is available [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

Now is the time to move back to the main procedure. First, the train data is fetched via the torchvision `torchvision.MNIST`. To be able to evaluate the network while it is being trained, we use a validation set. Here the train set is split into two disjoint sets: the training and the validation set using `torch.utils.data.random_split`.

7. Initialize the network with the `Net` object (see the `torch` documentation for help).

8. Train your network for a fixed number of `EPOCHS`. Epochs are the number of times the learning algorithm will work through the entire training dataset. Major steps in the training loop include normalizing inputs, model prediction, loss calculation, `.backward()` over loss to compute gradients, `sgd_step` to update model parameters and `zero_grad` to reset gradients. Validate the model once per epoch.
    
9. After the model is trained, load the test data with `test_loader` and calculate the test accuracy.

10. Optional: Plot the training and validation accuracies and add the test accuracy in the end.
