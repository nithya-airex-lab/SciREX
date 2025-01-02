from scirex.core.model_compression.pruning import ModelPruning
from scirex.core.model_compression.quantization import ModelQuantization

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the input images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Pruning Example
pruner = ModelPruning()
pruner.train_baseline_model(train_images, train_labels)
baseline_accuracy = pruner.evaluate_baseline(test_images, test_labels)
print("Baseline test accuracy:", baseline_accuracy)
baseline_model_path = pruner.save_baseline_model()
print("Baseline model saved at:", baseline_model_path)

# Quantization Example
quantizer = ModelQuantization()
quantized_model = quantizer.quantize_model(pruner.baseline_model)
quantized_model_path = quantizer.save_quantized_model("quantized_mnist_model.tflite")
print("Quantized model saved at:", quantized_model_path)

