from model.data_utils import NERDataset
from model.sal_blstm_oal_crf_model import SAL_BLSTM_OAL_CRF_Model
from model.blstm_crf_model import BLSTM_CRF_Model
from model.config import Config
import os
import sys

datasets = {"c":"conll2003", "o":"ontonotes-nw", "r":"ritter2011", "w":"wnut2016"}

def main(warmup=False):
    # create instance of config
    config = Config(load=False)

    source_dataset = "o"
    target_dataset = "r"
    config.batch_size = 10
    config.filename_train = "../datasets/%s/train_bioes"%datasets[target_dataset]
    config.filename_dev = "../datasets/%s/dev_bioes"%datasets[target_dataset]
    config.filename_test = "../datasets/%s/test_bioes"%datasets[target_dataset]

    # Enable the below line only when you are using different embs.
    # Make sure you have run the "python prep_data.py source_dataset target_dataset"
    # Make sure you have run the "python prep_data.py target_dataset target_dataset"
    config.filename_words = "../datasets/%s/words.txt"%datasets[source_dataset]
    config.filename_chars = "../datasets/%s/chars.txt"%datasets[source_dataset]
    config.filename_tags = "../datasets/%s/tags.txt"%datasets[target_dataset]
    config.filename_trimmed = config.filename_trimmed.replace("dataset_name",datasets[source_dataset])

    config.dir_model = config.dir_model.replace("/source", "/target")
    config.dir_output = config.dir_output.replace("/source", "/target")
    config.path_log = config.path_log.replace("/source", "/target")
    config.oal_hidden_size_lstm = 100
    config.psi = 1
    config.load()

    # create datasets
    train = NERDataset(config.filename_train, config.processing_word,
                       config.processing_tag, config.max_iter)

    dev   = NERDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    if config.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids[2])


    if warmup=="none":
        config.lr_method        = "adam"
        config.lr               = 0.001
        config.lr_decay         = 1
        config.batch_size = 10
        config.psi = 1
        config.nepochs = 50
        model = SAL_BLSTM_OAL_CRF_Model(config)
        model.build()
        model.restore_session("results/source/model.weights/", transfer_mode=True)
        model.train(train, dev)


if __name__ == "__main__":
    if len(sys.argv)==1 or sys.argv[1] == "none":
        main(warmup="none")
