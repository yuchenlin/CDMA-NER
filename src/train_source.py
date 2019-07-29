from model.data_utils import NERDataset
from model.blstm_crf_model import BLSTM_CRF_Model
from model.config import Config
import os


datasets = {"c":"conll2003", "o":"ontonotes-nw", "r":"ritter2011", "w":"wnut2016"}

def main():
    # create instance of config
    config = Config(load=False)

    source_dataset = "o"
    config.batch_size = 50
    config.nepochs = 10
    config.filename_train = "../datasets/%s/train_bioes"%datasets[source_dataset]
    config.filename_dev = "../datasets/%s/dev_bioes"%datasets[source_dataset]
    config.filename_test = "../datasets/%s/test_bioes"%datasets[source_dataset]

    config.filename_words = "../datasets/%s/words.txt"%datasets[source_dataset]
    config.filename_tags = "../datasets/%s/tags.txt"%datasets[source_dataset]
    config.filename_chars = "../datasets/%s/chars.txt"%datasets[source_dataset]

    config.filename_trimmed = config.filename_trimmed.replace("dataset_name",datasets[source_dataset])
    config.load()

    if config.gpu_ids:
        print("using gpu ids:", config.gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids[4])
    # build model
    model = BLSTM_CRF_Model(config)
    model.build()

    # create datasets
    train = NERDataset(config.filename_train, config.processing_word,
                       config.processing_tag, config.max_iter)

    dev   = NERDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
