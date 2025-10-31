import numpy as np

__all__ = ['majority_label', 'average_label']

"""
Returns the majority label from a list of labels.
Inputs:
    - labels: a list of labels
"""
def majority_label(labels):
    labels = np.array(labels)
    # check for empty
    if labels.size == 0:
        return None
    # get counts of labels
    label, count = np.unique(labels, True)
    # return max count
    max = np.argmax(count)
    max_labels = label[count == max]
    for label in labels:
        if label in max_labels:
            return label

"""
Returns the average of a list of labels.
Inputs:
    - labels: a list of labels
"""
def average_label(labels):
    labels = np.array(labels, np.float64)
    # check for empty
    if labels.size == 0:
        return np.nan
    # return average
    return np.mean(labels)