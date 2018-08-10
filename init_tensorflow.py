import os
import tensorflow as tf


def init_tensorflow(GPU='0', GPU_MEM_FRACTION=0.8):
    # init tensorflow session
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    config = tf.ConfigProto()
    # set gpu mem usage ratio
    config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    tf_session = tf.Session(config=config)
