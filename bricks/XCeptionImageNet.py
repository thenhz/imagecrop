import os 

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.cm as cm
from scipy import interpolate

class XCeptionImageNet:

    def __init__(
        self
        ):
        
        self.model = keras.applications.xception.Xception(weights="imagenet")
        self.last_conv_layer_name = "block14_sepconv2_act"
        self.classifier_layer_names = [
            "avg_pool",
            "predictions",
        ]

        #Setting up a model that returns the last convolutional output
        self.last_conv_layer = self.model.get_layer(self.last_conv_layer_name)
        self.last_conv_layer_model = keras.Model(self.model.inputs, self.last_conv_layer.output)

        #Setting up a model that goes from the last convolutional output to the final predictions
        self.classifier_input = keras.Input(shape=self.last_conv_layer.output.shape[1:])
        x = self.classifier_input
        for layer_name in self.classifier_layer_names:
            x = self.model.get_layer(layer_name)(x)
        self.classifier_model = keras.Model(self.classifier_input, x)

    def get_img_array(self, img_path, target_size):
            img = keras.preprocessing.image.load_img(img_path)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_resized = img.resize(target_size)
            array = keras.preprocessing.image.img_to_array(img_resized)
            array = np.expand_dims(array, axis=0)
            return array, img_array.shape[1], img_array.shape[0]
    
    def get_detections(self, image_path):
        img_array, orig_width, orig_height = self.get_img_array(image_path, target_size=(299, 299))
        #preds = self.model.predict(img_array)

        
        #Retrieving the gradients of the top predicted class with regard to the last convolutional output
        with tf.GradientTape() as tape:
            last_conv_layer_output = self.last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = self.classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        #Gradient pooling and channel importance weighting
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        heatmap = np.uint8(255 * heatmap)

        #just to perform a resize in the next step
        heatmap = np.expand_dims(heatmap, axis=-1)

        heatmap = keras.preprocessing.image.array_to_img(heatmap)
        heatmap = heatmap.resize((orig_width, orig_height))
        heatmap = keras.preprocessing.image.img_to_array(heatmap)

        #bring back to original rank
        heatmap = np.squeeze(heatmap, axis=-1)

        return heatmap