import time
import os


class Config:
    """
    Basic class for configurations
    """

    input_size = 27 #len(feature_columns)
    output_size = 27

    hidden_size = 128
    lstm_layers = 8
    dropout_rate = 0.2

    do_train = True
    do_predict = True
    add_train = False
    shuffle_train_data = False
    use_cuda = True

    train_data_rate = 0.85
    valid_data_rate = 0.15

    batch_size = 320
    learning_rate = 0.001
    epoch = 64

    random_seed = 42

    debug_mode = False
    debug_num = 666

    name = "pytorch_rttts_lstm"
    model_name = "model_" + name + ".pth"

    train_data_path = "./data/ringtones_cleaned.txt"
    model_save_path = "./checkpoint/"
    log_save_path = "./log/"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if do_train:
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + "/"
        os.makedirs(log_save_path)
