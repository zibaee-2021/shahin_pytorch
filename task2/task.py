"""
Task 2 A depth-wise separable convolution

For the purpose of the coursework, the dataset is only split into two, training and test sets.

• Adapt the Image Classification tutorial to use a different network, VisionTransformer. You can choose
any configuration that is appropriate for this application. [5]
o TensorFlow version
o PyTorch version
• Implement a data augmentation class MixUp, using the mixup algorithm, such that: [10]
o Inherited from the relevant classes in TensorFlow/PyTorch is recommended but not assessed.
o The MixUp algorithm can be applied to images and labels in each training iteration.
o Have an input flag “sampling_method” and appropriate hyperparameters for two options:
▪ sampling_method = 1: λ is sampled from a beta distribution as described in the paper.
▪ sampling_method = 2: λ is sampled uniformly from a predefined range.
▪ The algorithm should be seeded for reproducible results.
o Visualise your implementation, by saving to a PNG file “mixup.png”, a montage of 16 images
with randomly augmented images that are about to be fed into network training.
o Note: the intention of this task is to implement the augmentation class from scratch using
only TensorFlow/PyTorch basic API functions. Using the built-in data augmentation classes
may result in losing all relevant marks.
• Implement a task script “task.py”, under folder “task2”, completing the following: [15]
o Train a new VisionTransformer classification network with MixUp data augmentation, for
each of the two sampling methods, with 20 epochs.
o Save the two trained models and submit your trained models within the task folder.
o Report the test set performance in terms of classification accuracy versus the epochs.
o Visualise your results, by saving to a PNG file “result.png”, a montage of 36 test images with
printed messages clearly indicating the ground-truth and the predicted classes for each.


"""