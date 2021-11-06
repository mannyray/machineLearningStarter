import tensorflow as tf
import sys
print(tf.__version__)

export_dir = sys.argv[1]#saved_model dir
output_dir = sys.argv[2]

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.target_spec.supported_ops = [tf.float16]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

with open(output_dir+'/'+'model.tflite', 'wb') as f:
    f.write(tflite_model)
