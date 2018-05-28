from environment import Environment
import tensorflow as tf

class Agent:

    def __init__(self, env:Environment):
        self.env = env
        input_shape = [None, env.obs_shape[1:]]
        output_shape = [None, env.obs_shape[1]]
        # self._s = sess

        # build the graph
        self.input = tf.placeholder(tf.float32, shape=input_shape)

        conv1 = tf.layers.conv2d(self.input,
                                filters=3,
                                kernel_size=[1, 3],
                                strides=[1, 1],
                                activation=tf.nn.relu,)
        print(conv1)

        conv2 = tf.layers.conv2d(conv1,
                                filters=10,
                                kernel_size=[1, 28],
                                strides=[1, 1],
                                activation=tf.nn.relu,)
        print(conv2)

        conv2_flatten = tf.contrib.layers.flatten(conv2)
        print(conv2_flatten)

        logits = tf.contrib.layers.fully_connected(conv2_flatten, output_shape[1],
                                          activation_fn=None)
        self.output = tf.nn.softmax(logits)
        print(self.output)



    def decide_action(self, observation):
        return self.env.action_sample()

    def train_step(self, obs, acts, rewards):
        batch_feed = {
            self.input: obs,
            self.output: acts
        }
