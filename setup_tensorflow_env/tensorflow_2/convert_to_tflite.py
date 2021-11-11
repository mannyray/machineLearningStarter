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

'''
import tensorflow as tf
import sys

export_dir = sys.argv[1]#saved_model dir
output_dir = sys.argv[2]


converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(export_dir+"/saved_model.pb",input_shapes = {'normalized_input_image_tensor':[1,320,320,3]},input_arrays=['normalized_input_image_tensor'],output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'])
converter.allow_custom_ops=True
converter.optimizations =  [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open(output_dir+"/model.tflite","wb").write(tflite_model)
'''
