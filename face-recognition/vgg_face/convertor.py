import tensorflow as tf

saved_model_dir = "C:\\Users\\aniket\\Desktop\\part-iv-project\\face-recognition\\vgg_face\\saved_model"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()

open("C:\\Users\\aniket\\Desktop\\part-iv-project\\face-recognition\\vgg_face\\optimised_model", "wb").write(tflite_quant_model)