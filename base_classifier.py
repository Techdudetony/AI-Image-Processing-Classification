import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        """
        Initialize the GradCAM object.
        :param model: The pre-trained model
        :param classIdx: The class index to visualize
        :param layerName: The name of the target layer
        """
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        """
        Find the target layer automatically by searching for the last 4D layer.
        :return: Name of the target layer
        """
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        """
        Compute the Grad-CAM heatmap.
        :param image: Preprocessed image
        :param eps: Small value to prevent division by zero
        :return: Heatmap
        """
        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
        """
        Overlay the heatmap on the image.
        :param heatmap: Heatmap
        :param image: Original image
        :param alpha: Opacity of the heatmap
        :param colormap: Colormap to use
        :return: Tuple of heatmap and overlayed image
        """
        # Resize heatmap to match the image size
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Ensure the image is in the correct format
        image = image.astype("uint8")

        # Overlay the heatmap on the image
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return (heatmap, output)

def add_black_box(image, x, y, width, height):
    """
    Add a black box to occlude part of the image.
    :param image: Original image
    :param x: X-coordinate of the top-left corner of the box
    :param y: Y-coordinate of the top-left corner of the box
    :param width: Width of the box
    :param height: Height of the box
    :return: Image with black box
    """
    image_with_box = image.copy()
    image_with_box[y:y+height, x:x+width] = 0
    return image_with_box

def add_blur(image, x, y, width, height):
    """
    Add blur to occlude part of the image.
    :param image: Original image
    :param x: X-coordinate of the top-left corner of the region
    :param y: Y-coordinate of the top-left corner of the region
    :param width: Width of the region
    :param height: Height of the region
    :return: Image with blurred region
    """
    image_with_blur = image.copy()
    roi = image_with_blur[y:y+height, x:x+width]
    roi = cv2.GaussianBlur(roi, (51, 51), 0)
    image_with_blur[y:y+height, x:x+width] = roi
    return image_with_blur

def add_noise(image, x, y, width, height):
    """
    Add noise to occlude part of the image.
    :param image: Original image
    :param x: X-coordinate of the top-left corner of the region
    :param y: Y-coordinate of the top-left corner of the region
    :param width: Width of the region
    :param height: Height of the region
    :return: Image with noisy region
    """
    image_with_noise = image.copy()
    roi = image_with_noise[y:y+height, x:x+width]
    noise = np.random.randint(0, 50, roi.shape, dtype='uint8')
    roi = cv2.add(roi, noise)
    image_with_noise[y:y+height, x:x+width] = roi
    return image_with_noise

def classify_image(img_array):
    """
    Classify the image and return predictions.
    :param img_array: Preprocessed image array
    :return: Top-3 predictions
    """
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

def classify_and_visualize(image_path):
    """
    Classify an image and visualize the Grad-CAM heatmap along with occlusion techniques.
    :param image_path: Path to the input image
    """
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        original_predictions = classify_image(img_array)
        print("Original Image Predictions:")
        for i, (imagenet_id, label, score) in enumerate(original_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        original_image = image.load_img(image_path)
        original_image = image.img_to_array(original_image).astype("uint8")

        # Resize the original image to match model input if necessary
        original_image_resized = cv2.resize(original_image, (224, 224))

        # Apply occlusion techniques
        black_box_image = add_black_box(original_image_resized, 50, 50, 50, 50)
        blurred_image = add_blur(original_image_resized, 50, 50, 50, 50)
        noisy_image = add_noise(original_image_resized, 50, 50, 50, 50)

        # Prepare occluded images for classification
        black_box_array = preprocess_input(np.expand_dims(black_box_image, axis=0))
        blurred_array = preprocess_input(np.expand_dims(blurred_image, axis=0))
        noisy_array = preprocess_input(np.expand_dims(noisy_image, axis=0))

        black_box_predictions = classify_image(black_box_array)
        blurred_predictions = classify_image(blurred_array)
        noisy_predictions = classify_image(noisy_array)

        print("Black Box Predictions:")
        for i, (imagenet_id, label, score) in enumerate(black_box_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        print("Blurred Region Predictions:")
        for i, (imagenet_id, label, score) in enumerate(blurred_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        print("Noisy Region Predictions:")
        for i, (imagenet_id, label, score) in enumerate(noisy_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        # Display the results
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_image_resized.astype("uint8") / 255.0)
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.title("Black Box")
        plt.imshow(black_box_image.astype("uint8") / 255.0)
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.title("Blurred Region")
        plt.imshow(blurred_image.astype("uint8") / 255.0)
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.title("Noisy Region")
        plt.imshow(noisy_image.astype("uint8") / 255.0)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "basic_cat.jpg"
    classify_and_visualize(image_path)
