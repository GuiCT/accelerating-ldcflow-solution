import tensorflow as tf
from keras.losses import Loss

class MeanMaxError(Loss):
    def call(self, y_true, y_pred):
        # Calculate the element-wise absolute difference
        errors = tf.abs(y_true - y_pred)
        # Calculate the maximum error for each tensor in the batch
        max_errors_per_tensor = tf.reduce_max(errors, axis=(1, 2, 3))
        # Calculate the mean of the maximum errors across the batch
        mean_max_error = tf.reduce_mean(max_errors_per_tensor)
        return mean_max_error
