from environment import Environment
import tensorflow as tf

class Agent:

    def __init__(self, env:Environment, sess, lr=0.0005):

        self.sess = sess

        self.env = env
        input_shape = [None, *env.obs_shape[1:]]
        action_shape = [None, env.obs_shape[1]]
        self.output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')

        # build the graph
        self.X = tf.placeholder(tf.float32, shape=input_shape, name='state')

        conv1 = tf.layers.conv2d(self.X,
                                filters=5,
                                kernel_size=[1, 3],
                                strides=[1, 1],
                                activation=tf.nn.relu,)
        print(conv1)

        conv2 = tf.layers.conv2d(conv1,
                                filters=100,
                                kernel_size=[1, 28],
                                strides=[1, 1],
                                activation=tf.nn.relu,)
        print(conv2)

        conv2_flatten = tf.contrib.layers.flatten(conv2)
        print(conv2_flatten)

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(conv2_flatten, self.output_keep_prob)

        logits = tf.contrib.layers.fully_connected(conv2_flatten, action_shape[1],
                                          activation_fn=None)


        # Action (portfolio weight vector)
        self.action = tf.nn.softmax(logits)
        print(self.action)

        # 학습을 위한 뉴런들
        # future_price: t+1 시점의 상대가격(t시점 대비)
        self.future_price = tf.placeholder(tf.float32, shape=action_shape)
        # batch_size 기간에서 달성될 수 있는 Optimal 한 PV 변화량
        # self.optimal_gain_pv = tf.placeholder(tf.float32, shape=[None, 1])

        self.pv_gain_vector = tf.reduce_sum(self.action * self.future_price, axis=1)
        print(self.pv_gain_vector)
        self.gain_pv = -tf.reduce_mean(tf.log(self.pv_gain_vector))
        self.portfolio_value = tf.reduce_prod(self.pv_gain_vector)

        self.loss = self.gain_pv
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)


    def decide_action(self, obs, random=False):
        if random:
            return self.env.action_sample()
        else:
            return self.sess.run(self.action, feed_dict={
                                self.X: obs, self.output_keep_prob:1.0})


    def train_step(self, obs_b, future_price_b):
        batch_feed = {
            self.X: obs_b,
            self.future_price:future_price_b,
            self.output_keep_prob: 0.5
        }

        l, pv, _ = self.sess.run([self.loss, self.portfolio_value, self.train_op], feed_dict=batch_feed)
        print(l, pv)