import tensorflow as tf

import dresnet

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("recursions", 9, "Number of recursions")
flags.DEFINE_integer("residuals", 3, "Number of residuals in the feature extraction subnet")
flags.DEFINE_integer("scale", 2, "Input data scaling")
flags.DEFINE_float("max_value", 255.0, "Input normalization range")
flags.DEFINE_integer("channels", 1, "Input depth")
flags.DEFINE_string("test_dir", "", "Directory with test files")
flags.DEFINE_boolean("load_model", True, "Load saved model before start")
flags.DEFINE_string("model_name", "", "model name for save files and tensorboard log")
flags.DEFINE_integer("feature_num", 96, "Number of CNN Filters")
flags.DEFINE_boolean("upscale", False, "Whether to upscale the test image by a scale")


def main(_):
	if FLAGS.model_name is "":
		model_name = "model_F%d_D%d_R%d" % (FLAGS.feature_num, FLAGS.recursions, FLAGS.residuals)
	else:
		model_name = FLAGS.model_name
  
	model = dresnet.Dresnet(FLAGS, model_name=model_name, load_ckpt=FLAGS.load_model)
	model.run_sr(FLAGS.test_dir)


if __name__ == '__main__':
	tf.app.run()
