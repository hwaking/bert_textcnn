# encoding=utf8



class TextConfig():

    seq_length=128         # max length of sentence
    num_labels=10          # number of labels

    num_filters=128        # number of convolution kernel
    filter_sizes=[2, 3, 4]   # size of convolution kernel
    hidden_dim=128         # number of fully_connected layer units

    keep_prob=0.5          # dropout
    lr= 5e-5               # learning rate
    lr_decay= 0.9          # learning rate decay
    clip= 5.0              # gradient clipping threshold

    is_training=True       # is _training
    use_one_hot_embeddings=False  # use_one_hot_embeddings


    num_epochs=3           # epochs
    batch_size=32          # batch_size
    print_per_batch =200   # print result
    require_improvement=1000   # stop training if no inporement over 1000 global_step

    output_dir = './result'
    data_dir = './data'  #the path of input_data file
    vocab_file = './bert/pre_train_model/chinese_L-12_H-768_A-12/vocab.txt'  # the path of vocab file
    bert_config_file = './bert/pre_train_model/chinese_L-12_H-768_A-12/bert_config.json'  # the path of bert_cofig file
    init_checkpoint = './bert/pre_train_model/chinese_L-12_H-768_A-12/bert_model.ckpt'   # the path of bert model
