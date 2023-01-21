import time
import os


class Config:
    """
    Basic class for configurations
    """
    feature_columns = list(range(2, 9))
    label_columns = [4, 5]

    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 128
    lstm_layers = 2
    dropout_rate = 0.2
    time_step = 20

    do_train = True
    do_predict = True
    add_train = True
    shuffle_train_data = True
    use_cuda = True

    train_data_rate = 0.95
    valid_data_rate = 0.15

    batch_size = 64
    learning_rate = 0.001
    epoch = 20

    random_seed = 42

    debug_mode = False
    debug_num = 666

    name = "pytorch_rttts_lstm"
    model_name = "model_" + name + ".pth"

    train_data_path = "./data/ringtones_clean.txt"
    model_save_path = "./checkpoint/"
    log_save_path = "./log/"
    do_log_print_to_screen = True

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if do_train:
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + "/"
        os.makedirs(log_save_path)
