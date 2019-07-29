from model.config import Config
from model.data_utils import NERDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word
import sys

datasets = {"c":"conll2003", "o":"ontonotes-nw", "r":"ritter2011", "w":"wnut2016"}

def get_vocabs_from_dataset(dataset):

    filename_train = "../datasets/%s/train_bioes"%datasets[dataset]
    filename_dev = "../datasets/%s/dev_bioes"%datasets[dataset]
    filename_test = "../datasets/%s/test_bioes"%datasets[dataset]


    processing_word = get_processing_word(lowercase=True)
    # Generators
    dev   = NERDataset(filename_dev, processing_word)
    test  = NERDataset(filename_test, processing_word)
    train = NERDataset(filename_train, processing_word)

    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    return vocab_words, vocab_tags

def main():
    # get config and processing of words
    config = Config(load=False)

    # or ontonotes-nw if you like
    assert sys.argv[1] in datasets, "the source argument should be in {c/o/r/w}"
    source_dataset = sys.argv[1]
    source_vocab_words, source_vocab_tags = get_vocabs_from_dataset(source_dataset)
    print("Source word vocab size:", len(source_vocab_words))

    assert sys.argv[2] in datasets, "the target argument should be in {c/o/r/w}"
    target_dataset = sys.argv[2]
    target_vocab_words, _ = get_vocabs_from_dataset(target_dataset)
    print("Target word vocab size:", len(target_vocab_words))

    # Build Word and Tag vocab
    config.filename_words = "../datasets/%s/words.txt"%datasets[source_dataset]
    config.filename_tags = "../datasets/%s/tags.txt"%datasets[source_dataset]
    config.filename_chars = "../datasets/%s/chars.txt"%datasets[source_dataset]

    print("Source+Target word vocab size:", len((source_vocab_words | target_vocab_words)))
    vocab_glove = get_glove_vocab(config.filename_glove)
    vocab = (source_vocab_words | target_vocab_words) & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)
    print("Final word vocab size:", len(vocab))

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(source_vocab_tags, config.filename_tags)

    # Build and save char vocab
    vocab_chars = get_char_vocab((source_vocab_words | target_vocab_words))
    print("Final char vocab size:", len(vocab_chars))
    write_vocab(vocab_chars, config.filename_chars)

    # Trim Word Vectors
    vocab = load_vocab(config.filename_words)
    config.filename_trimmed = config.filename_trimmed.replace("dataset_name",datasets[source_dataset])
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)



if __name__ == "__main__":
    main()
