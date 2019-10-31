import tensorflow as tf


def configure_flags():
    tf.flags.DEFINE_string("run_name", "experiment1", "Directory for experiment logging")
    tf.flags.DEFINE_integer("batch_size", 32, "Number of images to be processed per batch")


def main(_):
    flags = tf.flags.FLAGS
    print(flags)


if __name__ == '__main__':
    configure_flags()
    tf.app.run()