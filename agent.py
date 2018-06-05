import tensorflow as tf


class Agent:

    def __init__(self, sess, obs_shape, lr=0.0005):
        """
        :param sess: tf.Session
        :param obs_shape: [1, nb_asset, window_size, nb_feature]
        :param lr: learning_rate
        """
        _, nb_asset, window_size, nb_feature = obs_shape
        input_shape = [None, nb_asset, window_size, nb_feature]
        action_shape = [None, nb_asset]

        self.sess = sess
        self.is_training = tf.placeholder(tf.bool)

        # build the graph
        self.X = tf.placeholder(tf.float32, shape=input_shape, name='observation')

        with tf.name_scope("Conv1"):
            conv1 = tf.layers.conv2d(self.X, filters=5, kernel_size=[1, 3], strides=[1, 1],
                                     activation=tf.nn.relu, )
        print(conv1)

        with tf.name_scope("Conv2"):
            conv2 = tf.layers.conv2d(conv1, filters=100, kernel_size=[1, window_size-3+1], strides=[1, 1],
                                     activation=tf.nn.relu, )
        print(conv2)

        conv2_flatten = tf.contrib.layers.flatten(conv2)
        print(conv2_flatten)

        # Add dropout
        with tf.name_scope("dropout"):
            conv2_flatten_drop = tf.layers.dropout(conv2_flatten, rate=0.5, training=self.is_training)

        logits = tf.contrib.layers.fully_connected(conv2_flatten_drop, action_shape[1],
                                                   activation_fn=None)

        # Action (portfolio weight vector)
        self.action = tf.nn.softmax(logits, name='action')
        print(self.action)

        # t시점 대비 t+1 시점의 상대가격 (price(t+1)/price(t))
        self.future_price = tf.placeholder(tf.float32, shape=action_shape, name='future_price')

        # TODO batch_size 기간에서 달성될 수 있는 Optimal 한 PV 변화량
        # 이를 이용해 loss 를 구성해 볼 생각임
        # self.optimal_gain_pv = tf.placeholder(tf.float32, shape=[None, 1])

        # t시점 대비 t+1 시점의 포트폴리오 가치 변화율.
        self.pv_gain_vector = tf.reduce_sum(self.action * self.future_price, axis=1)

        # batch 기간동안의 최종 포트폴리오 가치 변화율. e.g batch 기간동안 1.5배 가치 상승
        self.portfolio_value = tf.reduce_prod(self.pv_gain_vector)

        # Sharpe_Ratio를 구하기 위한 tensor들...
        # 배치 기간동안 마켓의 수익률
        self.mkt_return = tf.reduce_mean(self.future_price, axis=-1)
        self.mean_log_mkt_return = tf.reduce_mean(tf.log(self.mkt_return))
        self.mean_log_pv_return = tf.reduce_mean(tf.log(self.pv_gain_vector))
        self.std_log_pv_return = tf.sqrt(tf.reduce_mean((tf.log(self.pv_gain_vector) - self.mean_log_pv_return) ** 2))

        self.sharpe_ratio = (self.mean_log_pv_return - self.mean_log_mkt_return) / self.std_log_pv_return

        # Information Ratio를 구하기 위한 tensor들...
        self.log_pv_mkt_diff = tf.log(self.pv_gain_vector) - tf.log(self.mkt_return)
        self.mean_log_pv_mkt_diff = tf.reduce_mean(self.log_pv_mkt_diff)
        self.std_log_pv_mkt_diff = tf.sqrt(tf.reduce_mean((self.log_pv_mkt_diff - self.mean_log_pv_mkt_diff) ** 2))

        self.information_ratio = self.mean_log_pv_mkt_diff / self.std_log_pv_mkt_diff

        # self.loss = -self.sharpe_ratio
        self.loss = -self.information_ratio
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)



    def decide_action(self, obs):
        return self.sess.run(self.action, feed_dict={
            self.X: obs, self.is_training: False})

    def train_step(self, obs_b, future_price_b):

        batch_feed = {
            self.X: obs_b,
            self.future_price: future_price_b,
            self.is_training: True,
        }

        l, pv, _, ir = self.sess.run([self.loss, self.portfolio_value, self.train_op,
                                      self.information_ratio],
                                 feed_dict=batch_feed)
        print("loss:{:9.6f} PV:{:9.6f} IR:{:9.6f}".format(l, pv, ir))

        return l, pv, ir
