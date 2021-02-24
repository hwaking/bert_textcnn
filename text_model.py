import tensorflow as tf
from bert import modeling

class TextCNN(object):

    def __init__(self, config):
        '''获取超参数以及模型需要的传入的5个变量，input_ids，input_mask，segment_ids，labels，keep_prob'''
        self.config = config

        print('self.config.lr : ', self.config.lr)
        self.bert_config = modeling.BertConfig.from_json_file(self.config.bert_config_file)
        self.input_ids = tf.placeholder(tf.int64, shape=[None, self.config.seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int64, shape=[None, self.config.seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int64, shape=[None, self.config.seq_length], name='segment_ids')
        self.labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.cnn()



    def cnn(self):

        '''获取bert模型最后的token-level形式的输出(get_sequence_output)，将此作为embedding_inputs，作为卷积的输入'''
        with tf.name_scope('bert'):
            bert_model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.config.is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=self.config.use_one_hot_embeddings)

            embedding_inputs = bert_model.get_sequence_output() # shape:[batch_size, sequence_length, hidden_size]

        '''用三个不同的卷积核进行卷积和池化，最后将三个结果concat'''
        with tf.name_scope('conv'):
            pooled_outputs = []
            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)

            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size,reuse=False):
                    conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, filter_size, name='conv1d') # conv shape:[batch_size, seqlength-filter_size+1, num_filters]
                    print('conv.shape:', conv.shape)
                    pooled = tf.reduce_max(conv, reduction_indices=[1], name='gmp') # global max pooling layer, pooled shape: [batch_size, num_filters]
                    print('pooled.shape:', pooled.shape)
                    pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, 1)  # h_pool shape:[batch_size, num_filters*3]
            print('h_pool.shape:', h_pool.shape)

            outputs = tf.reshape(h_pool, [-1, num_filters_total])  # shape: [batch_size, num_filters*3]
            print('outputs.shape: ', outputs.shape)

        '''加全连接层和dropuout层'''
        with tf.name_scope('fc'):
            fc = tf.layers.dense(outputs, self.config.hidden_dim, name='fc1')  # shape:[batch_size, hidden_dim]
            print('fc.shape-dense:', fc.shape)
            fc = tf.nn.dropout(fc, self.keep_prob)  # shape:[batch_size, hidden_dim]
            print('fc.shape-dropout:', fc.shape)
            fc = tf.nn.relu(fc)  # shape:[batch_size, hidden_dim]
            print('fc.shape-relut:', fc.shape)

        '''logits: 每句话变成 一个分类'''
        with tf.name_scope('logits'):
            self.logits = tf.layers.dense(fc, self.config.num_labels, name='logits')  # shape:[batch_size, num_labels]
            print('self.logits: ', self.logits.shape, self.logits)  # [, 10]
            self.prob = tf.nn.softmax(self.logits)  # onehot, shape: [batch_size, num_labels]
            print('self.prob.shape: ', self.prob.shape)  # shape: [batch_size, num_labels]
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 真实标签id
            print('self.y_pred_cls.shape: ', self.y_pred_cls.shape)

        '''计算loss，因为输入的样本标签不是one_hot的形式，需要转换下
            loss = -∑∑p(Xij)*log(q(Xij)), q(Xij)为预测值，p(Xij)真实值
        '''
        with tf.name_scope('loss'):
            log_probs = tf.nn.log_softmax(self.logits, axis=-1)  # 相对于使用直接使用softmax，使用log_softmax 是为了防止数据下溢，是数据更稳定；
            one_hot_labels = tf.one_hot(self.labels, depth=self.config.num_labels, dtype=tf.float32)  # 将输入label 变为onehot 样式；
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # 单个样本交叉熵损失函数
            self.loss = tf.reduce_mean(per_example_loss) # 平均交叉熵损失

        '''optimizer'''
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)  # 优化器选择
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 损失函数对于变量的导数，返回梯度与变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)  # 梯度裁剪
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)  # 使用梯度更新变量（权重）

        '''accuracy'''
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.labels, self.y_pred_cls)
            self.acc=tf.reduce_mean(tf.cast(correct_pred, tf.float32))
