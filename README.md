Natural Language Processing Notebooks
--
### Written for Practicing Engineers

This work builds on the outstanding work which exists on Natural Language Processing. These range from classics like Jurafsky's Speech and Language Processing to rather modern work in The Deep Learning Book by Longfellow et al.

While they are great as introductory textbooks for college students - this is intended for practitioners to quickly read, skim, select what is useful and then proceed. There are several notebooks divided into 7 logical themes.

Each section builds on ideas and code from previous notebooks, but you can fill in the gaps mentally and jump directly to what interests you.

## Chapter 01 
[Introduction To Text Processing, with Text Classification](https://github.com/NirantK/nlp-python-deep-learning/blob/master/Part-01.ipynb)
- Perfect for Getting Started! We learn better with code-first approaches. 

## Chapter 02
- [Text Cleaning](https://github.com/NirantK/nlp-python-deep-learning/blob/master/Part-02-A.ipynb) notebook, code-first approaches with supporting explanation. Covers some simple ideas like:
  - Stop words removal
  - Lemmatization
- [Spell Correction](https://github.com/NirantK/nlp-python-deep-learning/blob/master/Part-02-B.ipynb) covers **almost everything** that you will ever need to get started with spell correction, similar words problems and so on. 

## Chapter 03
[Leveraging Linguistics](https://github.com/NirantK/nlp-python-deep-learning/blob/master/Part-03%20NLP%20with%20spaCy%20and%20Textacy.ipynb) is an important toolkit in any practitioners toolkit. Using **spaCy** and textacy we look at two interesting challenges and how to tackle them: 
- Redacting names 
  - Named Entity Recognition
- Question and Answer Generation
  - Part of Speech Tagging
  - Dependency Parsing

## Chapter 04
- [Text Representations](https://github.com/NirantK/nlp-python-deep-learning/blob/master/Part-04%20Text%20Representations.ipynb) is about converting text to numerical representations aka vectors. 
- Covers popular celebrities: word2vec, fasttext and doc2vec - document similarity using the same
- Hands on Guide to **gensim**

## Chapter 05
[Modern Methods for Text Classification](https://github.com/NirantK/nlp-python-deep-learning/blob/master/Part-05%20Modern%20Text%20Classification.ipynb). 
- Simple Classifiers and How to Optimize Them from **scikit-learn**
- How to combine and **ensemble** them for increased performance
- Builds intuition for ensembling - so that you can write your own ensembling techniques

## Chapter 06
[Deep Learning for NLP](https://github.com/NirantK/nlp-python-deep-learning/blob/master/Part-06%20Deep%20Learning%20for%20NLP.ipynb) 
- From scratch code tutorial with Text Classification as an example
- Using **PyTorch** and *torchtext*
- Write our own data loaders, pre-processing, training loop and other utilities
