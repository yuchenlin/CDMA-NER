## CDMA-NER: Cross-Domain Model Adaptation for Named Entity Recognition 
CDMA-NER is an archived codebase for cross-domain named entity recognition based on our proposed *neural adaptation layers*. 
The major advantages of the CDMA-NER are as follows: 
- being lightweight (a minimum need for retraining, once you have a ready pre-trained source model), 
- exploiting the power of domain-specific word embeddings while improving the transferability.
- better performance than simple transfer learning techniques like fine-tuning and multi-task learning. 

Simply put, the CDMA-NER needs a pre-trained source model trained with word embeddings from a source-domain corpus. 
We first train a word adaptation layer based on the word frequency lists of the source and  target corpus as well as optional "pivot lexicon". 
Then, just feed a relatively smaller training data set to the interested target model with target-specific label space.
The target model in CDMA-NER has a sentence-level and a output-level adaptation layer addressing domain shift at the different levels respectively.
Note that a great feature of CDMA-NER is that you can use a totally different sets of entity types.



### Tutorial

All the code are runnable under python 3.5.2 and tensorflow r1.14, while other common environment should be okay as well. When the necessary material are ready, you can run the code step by step as follows. Make sure you have prepared your source and target corpora, datasets and pre-trained word embeedeings as well as optional pivot lexicon if you need.

#### - Pre-processing the data 

- For word adaptation, you can first run the codes in `src/word_adapt` to get the pivot lexicon based on intersection of source&target domain corpora. We also include a simple linear transformation to obtain cross-domain word embeddings for target domain, while we suggest run [CCA](https://github.com/mfaruqui/crosslingual-cca) to get that. 

- Building char, tag, word sets: 

```
cd src/
python prep_data.py [source_dataset_name] [target_dataset_name]
python prep_data.py [target_dataset_name] [target_dataset_name]
```

After this, you should have a `data` folder at the same level containing datasets/[dataset_name]/[tags/words/chars].txt for further computation in the following steps. Note that you have to put the source and target domain-specific word embeddings as following the settings in model/config.py.

#### - Training the Source Model
Running
```
python train_source.py
``` 
will give you a trained source model based on BLSTM_CRF for source NER data set. You can change the path to the source corpus and embeddings in the `prep_source_data.py` and `train_source.py`. The model training will stop when the performance is not increasing up to 5 epochs, which you can change in the `model/config.py`. 

#### - Transfer Learning the Target Model
Finally, you can now run
 ```
 python transfer2target.py
 ``` 
to transfer the source model by fine-tuning it with data in target domain. The model path is saved at ``results/target/model.weigths/``. 


#### - Tagging & Evaluation 

For tagging new data or evaluating the results on test data, you can follow the scripts in ``evaluate.py``. 


 

### Transferring newswire data to social media.

#### Resource 
- Source Domain (Newswire)
    - [CoNLL 2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003)
    - [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
    - source corpus ([NYT + DailyMail](#))
    
- Target Domain (Social Media)
    - [Ritter 2011](https://github.com/aritter/twitter_nlp/blob/master/data/annotated/ner.txt)
    - [WNUT 2016](https://github.com/aritter/twitter_nlp/tree/master/data/annotated/wnut16)
    - target corpus ([Twitter Stream Grab](https://archive.org/details/twitterstream)) 
    - [Twitter Normalization Lexicon](http://www.hlt.utdallas.edu/~yangl/data/Text_Norm_Data_Release_Fei_Liu/) (Liu et al. ACL 2012)
        
- General Domain (only used for ablation study. Note that this is not the trained embedding on both source and target corpora, which is the case in the paper.)
    - [general_emb](https://nlp.stanford.edu/projects/glove/) 
    

Following the above steps with such resources, you can replicate our experiments in the paper. 
For more details and experiments, please refer to our paper "Neural Adaptation Layers for Cross-domain Named Entity Recognition" (Lin and Lu, EMNLP 2018) and the supplementary material. 
```
@InProceedings{bylin-18-nalner,
  author    = {Lin, Bill Yuchen and Lu, Wei},
  title     = {Neural Adaptation Layers for Cross-domain Named Entity Recognition},
  booktitle = {Proceedings of the Conference on Empirical Methods on Natural Language Processing (EMNLP)},
  month     = {November},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
}
``` 

Some of the pre-processing and general model utils are inspired by [the blog](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html). 
