import tensorflow as tf


class Agent:

    def __init__(self, name, sess, obs_shape, lr=0.000005):
        """
        :param sess: tf.Session
        :param obs_shape: [1, nb_asset, window_size, nb_feature] e.g. [1, 15, 30, 5]
        :param lr: learning_rate
        """
        _, nb_asset, window_size, nb_feature = obs_shape
        input_shape = [None, nb_asset, window_size, nb_feature]
        action_shape = [None, nb_asset]

        self.name = name
        self.sess = sess
        self.is_training = tf.placeholder(tf.bool)
        self.batch_size = tf.placeholder(tf.int32)
        self._global_step = 0
        self.global_step = tf.placeholder(tf.int64)
        # build the graph
        self.X = tf.placeholder(tf.float32, shape=input_shape, name='observation')

        with tf.name_scope("Conv1"):
            conv1 = tf.layers.conv2d(self.X, filters=128, kernel_size=[1, 3], strides=[1, 1],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     activation=tf.nn.relu)
            print(conv1)
            conv1 = tf.layers.max_pooling2d(conv1, [1, 2], [1, 2])
            print(conv1)
        conv1 = tf.layers.dropout(conv1, rate=0.3, noise_shape=[self.batch_size, nb_asset, (window_size-3+1)//2, 128], training=self.is_training)

        # conv1 = tf.transpose(conv1, [0, 2, 1, 3])
        # print(conv1)
        # conv1 = tf.reshape(conv1, [-1, window_size-3+1, 15 * 128])

        # cells = [tf.contrib.rnn.GRUCell(128) for _ in range(2)]
        # cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        #
        # rnn_out, _states = tf.nn.dynamic_rnn(cells, conv1, dtype=tf.float32)
        #
        # print(rnn_out)

        with tf.name_scope("Conv2"):
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=[1, (window_size-3+1)//2], strides=[1, 1],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     activation=tf.nn.relu)
        print(conv2)
        conv2 = tf.layers.dropout(conv2, rate=0.3, noise_shape=[self.batch_size, nb_asset, 1, 64],training=self.is_training)

        with tf.name_scope("Conv3"):
            conv3 = tf.layers.conv2d(conv2, filters=1, kernel_size=[1, 1], strides=[1, 1],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

        # conv2_flatten = tf.contrib.layers.flatten(rnn_out)
        # print(conv2_flatten)

        # logits = tf.contrib.layers.fully_connected(conv2_flatten, action_shape[1],
        #                                            activation_fn=None)
        conv3 = tf.contrib.layers.flatten(conv3)

        # Action (portfolio weight vector)
        self.action = tf.nn.softmax(conv3, name='action')
        print(self.action)


        self.mean_action = 1 / nb_asset
        print(self.mean_action)
        self.std_action = tf.sqrt(tf.reduce_mean((self.mean_action - self.action) ** 2, axis=-1))
        self.mean_std_action = tf.reduce_mean(self.std_action)
        print(self.std_action)
        print(self.mean_std_action)

        # t시점 대비 t+1 시점의 상대가격 (price(t+1)/price(t))
        self.future_price = tf.placeholder(tf.float32, shape=action_shape, name='future_price')

        # t시점 대비 t+1 시점의 포트폴리오 가치 변화율.
        self.pv_gain_vector = tf.reduce_sum(self.action * self.future_price, axis=1)

        # batch 기간동안의 최종 포트폴리오 가치 변화율. e.g batch 기간동안 1.5배 가치 상승
        self.portfolio_value = tf.reduce_prod(self.pv_gain_vector)

        # t시점 대비 t+1 시점의 마켓 가치 변화율.
        self.mkt_return = tf.reduce_mean(self.future_price, axis=-1)

        # Sharpe_Ratio를 구하기 위한 tensor들...
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
        # self.reward = self.information_ratio - 0.1 * self.mean_std_action
        # self.reward = self.information_ratio
        self.reward = self.mean_log_pv_mkt_diff
        # self.reward = self.sharpe_ratio
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(-self.reward)
        # self.train_op = tf.train.AdagradOptimizer(learning_rate=lr).minimize(-self.reward)
        # self.train_op = tf.train.AdagradDAOptimizer(learning_rate=lr, global_step=self.global_step).minimize(-self.reward)
        # self.train_op = tf.train.A

    def decide_action(self, obs):
        return self.sess.run(self.action, feed_dict={
            self.X: obs, self.is_training: False, self.batch_size: len(obs)})

    def run_batch(self, obs_b, future_price_b, is_train, verbose=False):

        batch_feed = {
            self.X: obs_b,
            self.future_price: future_price_b,
            self.is_training: is_train,
            self.batch_size: len(obs_b),
            self.global_step: self._global_step
        }

        if is_train:
            self._global_step += 1
            rw, pv, _, ir, pv_vec, mean_std_act, actions = self.sess.run([self.reward, self.portfolio_value, self.train_op,
                                          self.information_ratio, self.pv_gain_vector, self.mean_std_action, self.action],
                                         feed_dict=batch_feed)
        else:
            rw, pv, ir, pv_vec, mean_std_act, actions = self.sess.run([self.reward, self.portfolio_value,
                                       self.information_ratio, self.pv_gain_vector, self.mean_std_action, self.action],
                                      feed_dict=batch_feed)

        if verbose:
            print("reward:{:9.6f} PV:{:9.6f} IR:{:9.6f} {}".format(rw, pv, ir, mean_std_act))

            if not is_train:
                print("action:")
                for act in actions:
                    print('test', end=' ')
                    for a in act:
                        print("{:6.2f}".format(a * 100), end=' ')
                    print()
            else:
                print("action:")
                for act in actions:
                    print('train', end=' ')
                    for a in act:
                        print("{:6.2f}".format(a * 100), end=' ')
                    print()
        return rw, pv, ir, pv_vec
