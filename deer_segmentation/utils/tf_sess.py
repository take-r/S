def create_session(gpu_id=1, allow_growth=True):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    tfconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=str(gpu_id),
                                                        allow_growth=True))
    set_session(tf.Session(config=tfconfig))

def clear_session():
    import keras.backend as K
    K.clear_session()