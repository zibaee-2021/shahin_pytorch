"""
Task 3 Ablation Study

Using the Image Classification tutorial, this task investigates the impact of the following modification to
the original network. To evaluate a modification, an ablation study can be used by comparing the
performance before and after the modification.
• Difference between training with the two λ sampling methods in Task 2.
• Implement a task script “task.py”, under folder “task3”, completing the following: [30]
o Random split the data into development set (80%) and holdout test set (20%).
o Random split the development set into train (90%) and validation sets (10%).
o Design at least one metric, other than the loss, on validation set, for monitoring during
training.
o Train two models using the two different sampling methods.
o Report a summary of loss values, speed, metric on training and validation.
o Save and submit these two trained models within the task folder.
o Report a summary of loss values and the metrics on the holdout test set. Compare the results
with those obtained during development.


"""