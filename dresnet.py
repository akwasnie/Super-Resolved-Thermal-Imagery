import os

import numpy as np
import tensorflow as tf

from deeply_recursive_cnn_tf import super_resolution_utilty as util


class Dresnet:
    def __init__(self, flags, model_name="", load_ckpt=True):
        
        self.recursions = flags.recursions
        self.residuals = flags.residuals
        self.scale = flags.scale
        self.max_value = flags.max_value
        self.channels = flags.channels
        self.test_dir = flags.test_dir
        self.load_model = flags.load_model
        self.model_name = flags.model_name
        self.feature_num = flags.feature_num
        self.upscale = flags.upscale

        self.sess = tf.InteractiveSession(config=tf.ConfigProto())
        self.rec_out = [0] * (self.recursions + 1)
        
        self.build_graph()
        self.init_variables()
   

    def batch_norm(self, inputs, data_format):
        return tf.layers.batch_normalization(
            inputs=inputs,
            axis=1 if data_format == 'channels_first' else 3,
            momentum=0.9,
            epsilon=1e-5,
            center=True,
            scale=True,
            training=True,
            fused=True)
            
        
    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, self.channels], name="X")
        self.y = tf.placeholder(tf.float32, shape=[None, None, None, self.channels], name="Y")

        with tf.variable_scope("W-1_conv"):
            self.W1_conv_FE = util.weight([3, 3, self.channels, self.feature_num], stddev=0.001, name="conv_W", initializer="he")
            self.B1_conv_FE = util.bias([self.feature_num], name="conv_B")
            H1_conv_FE = util.conv2d_with_bias(self.x, self.W1_conv_FE, 1, self.B1_conv_FE, add_relu=False, name="H")
            H1_conv_FE = tf.nn.relu(H1_conv_FE)
            skip1=H1_conv_FE
           
        self.W2_conv_FE = util.weight([3, 3, self.feature_num, self.feature_num], stddev=0.001, name="conv_W", initializer="he")
        self.B2_conv_FE = util.bias([self.feature_num], name="conv_B")
        self.W3_conv_FE = util.weight([3, 3, self.feature_num, self.feature_num], stddev=0.001, name="conv_W", initializer="he")
        self.B3_conv_FE = util.bias([self.feature_num], name="conv_B")
        H_tmp_conv_FE = H1_conv_FE

        for block_iter in range(self.residuals):
            with tf.variable_scope("W-a-{0}_conv".format(1+block_iter)):
                H2_conv_FE = util.conv2d_with_bias(H_tmp_conv_FE, self.W2_conv_FE, 1, self.B2_conv_FE, add_relu=False, name="H")
                H2_conv_FE = self.batch_norm(H2_conv_FE, "channels_last")
                H2_conv_FE = tf.nn.relu(H2_conv_FE)

            with tf.variable_scope("W-b-{0}_conv".format(1+block_iter)):
                H3_conv_FE = util.conv2d_with_bias(H2_conv_FE, self.W3_conv_FE, 1, self.B3_conv_FE, add_relu=False, name="H")
                H3_conv_FE = self.batch_norm(H3_conv_FE, "channels_last")

            with tf.variable_scope("res{0}".format(1+block_iter)):
                H3_conv_FE += skip1
                H3_conv_FE = tf.nn.relu(H3_conv_FE)
                H_tmp_conv_FE = H3_conv_FE

        with tf.variable_scope("W0_conv"):
            self.W0_conv_FE = util.weight([3, 3, self.feature_num, self.feature_num], stddev=0.001, name="conv_W", initializer="he")
            self.B0_conv_FE = util.bias([self.feature_num], name="conv_B")
            self.rec_out[0] = util.conv2d_with_bias(H_tmp_conv_FE, self.W0_conv_FE, 1, self.B0_conv_FE, add_relu=False, name="H")
            self.rec_out[0] = tf.nn.relu(self.rec_out[0])
    

        self.W_conv_EM = util.weight([3, 3, self.feature_num, self.feature_num], stddev=0.001, name="W_conv", initializer="he")
        self.B_conv_EM = util.bias([self.feature_num], name="B")

        for i in range(0, self.recursions):
            with tf.variable_scope("W%d_conv" % (i+1)):
                Hm_temp2_conv_EM = self.rec_out[i]
                Hm_temp2_conv_EM = util.conv2d_with_bias(Hm_temp2_conv_EM, self.W_conv_EM, 1, self.B_conv_EM, add_relu=False, name="H%d" % i)
                Hm_temp2_conv_EM = tf.nn.relu(Hm_temp2_conv_EM)
                self.rec_out[i + 1] = Hm_temp2_conv_EM

        self.WD1_conv_REC = util.weight([3, 3, self.feature_num, self.feature_num], stddev=0.001, name="WD1_conv", initializer="he")
        self.BD1_conv_REC = util.bias([self.feature_num], name="BD1")

        self.WD2_conv_REC = util.weight([3, 3, self.feature_num+1, self.channels], stddev=0.001, name="WD2_conv", initializer="he")
        self.BD2_conv_REC = util.bias([1], name="BD2")

        self.Y1_conv = [0] * (self.recursions)
        self.Y2_conv = [0] * (self.recursions)
        self.W = tf.Variable(np.full(fill_value=1.0 / self.recursions, shape=[self.recursions], dtype=np.float32), name="LayerWeights")
        W_sum = tf.reduce_sum(self.W)

        self.y_outputs = [0] * self.recursions

        for i in range(0, self.recursions):
            with tf.variable_scope("Y%d" % (i+1)):
                self.Y1_conv[i] = util.conv2d_with_bias(self.rec_out[i+1], self.WD1_conv_REC, 1, self.BD1_conv_REC, add_relu=True, name="conv_1")
                y_conv = tf.concat([self.Y1_conv[i], self.x], 3)
                self.Y2_conv[i] = util.conv2d_with_bias(y_conv, self.WD2_conv_REC, 1, self.BD2_conv_REC, add_relu=True, name="conv_2")
                self.y_outputs[i] = self.Y2_conv[i] * self.W[i] / W_sum
                
        self.y_ = tf.add_n(self.y_outputs)

    
    def init_variables(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if self.load_model:
            self.saver.restore(self.sess, self.model_name + ".ckpt")
            print("Model restored.")


    def run_sr(self, test_dir, image_extension='.png'):
        out_dir = os.path.join('output')
        util.make_dir(out_dir)
        images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(image_extension)]
        for img in images:
            print('Processing:', img)
            file_path = img
            filename, extension = os.path.splitext(file_path)
            org_image = util.load_image(file_path)
            if len(org_image.shape) == 2:
                org_image = org_image.reshape(org_image.shape[0], org_image.shape[1], 1)
            
            if len(org_image.shape) >= 3 and org_image.shape[2] == 3:
                util.save_image(os.path.join(out_dir, filename + "_bicubic" + extension), org_image)
                ycbcr = util.convert_rgb_to_ycbcr(org_image)
                ycbcr = ycbcr.reshape(1, *ycbcr.shape)
                out_y = self.sess.run(self.y_, feed_dict={self.x: ycbcr[:, :, 0:1]})[0]
                util.save_image(os.path.join(out_dir, filename + "_result_y" + extension), out_y)
                image = util.convert_y_and_cbcr_to_rgb(out_y, ycbcr[:, :, 1:3])
            else:
                util.save_image(os.path.join(out_dir, filename + "_bicubic" + extension), org_image)
                if self.upscale:
                    org_image = util.resize_image_by_pil_bicubic(org_image, self.scale)
                scaled_image = org_image.reshape(1, *org_image.shape)
                image = self.sess.run(self.y_, feed_dict={self.x: scaled_image})[0]
            util.save_image(os.path.join(out_dir, filename + "_result" + extension), image)
        return
