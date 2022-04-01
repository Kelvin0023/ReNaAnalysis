import tensorflow as tf
import tempfile
import os

def get_model_size(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()

    # Create float TFLite model.
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite_model = float_converter.convert()

    quantized_tflite_model = converter.convert()

    # Create float TFLite model.
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite_model = float_converter.convert()

    # Measure sizes of models.
    _, float_file = tempfile.mkstemp('.tflite')
    _, quant_file = tempfile.mkstemp('.tflite')

    with open(quant_file, 'wb') as f:
      f.write(quantized_tflite_model)

    with open(float_file, 'wb') as f:
      f.write(float_tflite_model)

    print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
    print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))