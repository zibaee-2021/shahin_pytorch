"""
Task 1 Stochastic Minibatch Gradient Descent for Linear Models

Implement a polynomial function polynomial_fun, that takes two input arguments, a weight vector 𝐰
of size 𝑀 + 1 and an input scalar variable 𝑥, and returns the function value 𝑦.
The polynomial_fun should be vectorised for multiple pairs of scalar input and output,
with the same 𝐰. [5]

Using the linear algebra modules in TensorFlow/PyTorch, implement a least square solver for fitting
the polynomial functions, fit_polynomial_ls, which takes 𝑁 pairs of 𝑥 and target values 𝑡 as input, with
an additional input argument to specify the polynomial degree 𝑀, and returns the optimum weight
vector hat_𝐰 ̂in least-square sense, i.e. ‖𝑡 − 𝑦‖2 is minimised. [5]

Using relevant functions/modules in TensorFlow/PyTorch, implement a stochastic minibatch gradient
descent algorithm for fitting the polynomial functions, fit_polynomial_sgd, which has the same input
arguments as fit_polynomial_ls does, with additional two input arguments, learning rate and
minibatch size. This function also returns the optimum weight vector hat_𝐰. During training, the function
should report the loss periodically using printed messages. [5]

Implement a task script “task.py”, under folder “task1”, performing the following: [15]

o Use polynomial_fun (𝑀 = 2, 𝐰 = [1,2,3]T) to generate a training set and a test set, in the
form of respectively and uniformly sampled 20 and 10 pairs of 𝑥, 𝑥𝜖[−20, 20], and 𝑡. The
observed 𝑡 values are obtained by adding Gaussian noise (standard deviation being 0.5) to 𝑦.
o Use fit_polynomial_ls (𝑀𝜖{2,3,4}) to compute the optimum weight vector hat_w using the training set.
For each 𝑀, compute the predicted target values 𝑦 ̂ for all 𝑥 in both the training
and test sets.

o Report, using printed messages, the mean (and standard deviation) in difference a) between
the observed training data and the underlying “true” polynomial curve; and b) between the
“LS-predicted” values and the underlying “true” polynomial curve.

o Use fit_polynomial_sgd (𝑀𝜖{2,3,4}) to optimise the weight vector 𝐰
̂ using the training set.
For each 𝑀, compute the predicted target values 𝑦
̂ for all 𝑥 in both the training and test sets.

o Report, using printed messages, the mean (and standard deviation) in difference between the
“SGD-predicted” values and the underlying “true” polynomial curve.

o Compare the accuracy of your implementation using the two methods with ground-truth on
test set and report the root-mean-square-errors (RMSEs) in both 𝐰 and 𝑦 using printed
messages.

o Compare the speed of the two methods and report time spent in fitting/training (in seconds)
using printed messages.
• Implement a task script “task1a.py”, under folder “task1”. [10]

o Experiment how to make 𝑀 a learnable model parameter and using SGD to optimise this more
flexible model.
o Report, using printed messages, the optimised 𝑀 value and the mean (and standard deviation) in
difference between the model-predicted values and the underlying “true” polynomial curve.
"""