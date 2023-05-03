import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging    # first of all import the module
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
from tensorflow import keras


import warnings
warnings.filterwarnings("ignore")



class CustomModel(keras.Model):
    def __init__(self, student,teacher):
        super(CustomModel, self).__init__()
        self.student = student
        self.teacher = teacher


    @property
    def metrics(self):
        metrics = super().metrics
        return metrics
    def compile(self, optimizer,loss_fn, metrics,alpha):
        self.alpha = alpha
        super().compile(optimizer=optimizer, metrics=metrics, loss=loss_fn)


    def train_step(self, data):
        x, y = data

        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            y_pred = self.student(x, training=True)  # Forward pass
            # Compute the loss value.

            hard_loss = self.compiled_loss(y,y_pred,regularization_losses=self.losses,)
            soft_loss = self.compiled_loss(teacher_predictions,y_pred,regularization_losses=self.losses,)
            loss=self.alpha*soft_loss+(1-self.alpha)*hard_loss


        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)

        results={m.name: m.result() for m in self.student.metrics}
        results.update({m.name: m.result() for m in self.student.metrics})
        return results

    def test_step(self, data):
        x, y = data
        # Calculate predictions
        y_pred = self.student(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
