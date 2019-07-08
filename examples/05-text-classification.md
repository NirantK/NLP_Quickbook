
# Modern Methods for Text Classification


```python
from pathlib import Path
import pandas as pd
import gzip
from urllib.request import urlretrieve
from tqdm import tqdm
import os
import numpy as np
# if you are using the fastAI environment, all of these imports work
```


```python
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)
```


```python
def get_data(url, filename):
    """
    Download data if the filename does not exist already
    Uses Tqdm to show download progress
    """
    if not os.path.exists(filename):

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename, reporthook=t.update_to)
```


```python
# Let's download some data:
data_url = 'http://files.fast.ai/data/aclImdb.tgz'
# get_data(data_url, 'data/imdb.tgz')
```

Before we proceed, *manually extract the files* please!
The *.tgz* extension is equivalent to *.tar.gz* here. 

On Windows, you might need a software like *7z* 
On Linux, you can probably use *tar -xvcf imdb.tgz* 


```python
data_path = Path(os.getcwd())/'data'/'aclImdb'
assert data_path.exists()
```

This is to check that we have extracted the files at the correct location


```python
for pathroute in os.walk(data_path):
    next_path = pathroute[1]
    for stop in next_path:
        print(stop)
```

    test
    train
    all
    neg
    pos
    all
    neg
    pos
    unsup



```python
train_path = data_path/'train'
test_path = data_path/'test'
```


```python
def read_data(dir_path):
    """read data into pandas dataframe"""
    
    def load_dir_reviews(reviews_path):
        files_list = list(reviews_path.iterdir())
        reviews = []
        for filename in files_list:
            f = open(filename, 'r', encoding='utf-8')
            reviews.append(f.read())
        return pd.DataFrame({'text':reviews})
        
    
    pos_path = dir_path/'pos'
    neg_path = dir_path/'neg'
    
    pos_reviews, neg_reviews = load_dir_reviews(pos_path), load_dir_reviews(neg_path)
    
    pos_reviews['label'] = 1
    neg_reviews['label'] = 0
    
    merged = pd.concat([pos_reviews, neg_reviews])
    df = merged.sample(frac=1.0) # shuffle the rows
    df.reset_index(inplace=True) # don't carry index from previous
    df.drop(columns=['index'], inplace=True) # drop the column 'index' 
    return df
```


```python
train_path = data_path/'train'
test_path = data_path/'test'
```


```python
%%time
train = read_data(train_path)
test = read_data(test_path)
```

    Wall time: 24.6 s



```python
test[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Well, I'm a few days late but what the hell......</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Just watched this and it was amazing. Was in s...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A hundred miles away from the scene of a grizz...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>This is a case where the script plays with the...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Final Draft - A screenwriter (James Van Der Be...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# test.to_csv(data_path/'test.csv', index=False)
```


```python
# train.to_csv(data_path/'train.csv', index=False)
```


```python
X_train, y_train = train['text'], train['label']
X_test, y_test = test['text'], test['label']
```


```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
```

## Logistic Regression


```python
from sklearn.linear_model import LogisticRegression as LR
```


```python
lr_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',LR())])
```


```python
%%time
lr_clf.fit(X=X_train, y=y_train) # note that .fit function calls are inplace, and the Pipeline is not re-assigned
```

    Wall time: 5.82 s





    Pipeline(memory=None,
         steps=[('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip...ty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False))])




```python
lr_predicted = lr_clf.predict(X_test)
```


```python
lr_acc = sum(lr_predicted == y_test)/len(lr_predicted)
lr_acc
```




    0.88316




```python
def imdb_acc(pipeline_clf):
    predictions = pipeline_clf.predict(X_test)
    assert len(y_test) == len(predictions)
    return sum(predictions == y_test)/len(y_test), predictions
```

### Remove Stop Words


```python
lr_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf',LR())])
lr_clf.fit(X=X_train, y=y_train)
lr_acc, lr_predictions = imdb_acc(lr_clf)
lr_acc
```




    0.879



### Increase the Ngram Range


```python
lr_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1,3))), ('tfidf', TfidfTransformer()), ('clf',LR())])
lr_clf.fit(X=X_train, y=y_train)
lr_acc, lr_predictions = imdb_acc(lr_clf)
lr_acc
```




    0.866



# Multinomial Naive Bayes


```python
from sklearn.naive_bayes import MultinomialNB as MNB
mnb_clf = Pipeline([('vect', CountVectorizer()), ('clf',MNB())])
```


```python
mnb_clf.fit(X=X_train, y=y_train)
mnb_acc, mnb_predictions = imdb_acc(mnb_clf)
mnb_acc
```




    0.81356



### Add TF-IDF


```python
mnb_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',MNB())])
mnb_clf.fit(X=X_train, y=y_train)
mnb_acc, mnb_predictions = imdb_acc(mnb_clf)
mnb_acc
```




    0.82956



### Remove Stop Words


```python
mnb_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf',MNB())])
mnb_clf.fit(X=X_train, y=y_train)
mnb_acc, mnb_predictions = imdb_acc(mnb_clf)
mnb_acc
```




    0.82992



### Add Ngram Range from 1 to 3


```python
mnb_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1,3))), ('tfidf', TfidfTransformer()), ('clf',MNB())])
mnb_clf.fit(X=X_train, y=y_train)
mnb_acc, mnb_predictions = imdb_acc(mnb_clf)
mnb_acc
```




    0.8572



### Change Fit Prior to False


```python
mnb_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1,3))), ('tfidf', TfidfTransformer()), ('clf',MNB(fit_prior=False))])
mnb_clf.fit(X=X_train, y=y_train)
mnb_acc, mnb_predictions = imdb_acc(mnb_clf)
mnb_acc
```




    0.8572



### Support Vector Machine


```python
from sklearn.svm import SVC
svc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',SVC())])
svc_clf.fit(X=X_train, y=y_train)
svc_acc, svc_predictions = imdb_acc(svc_clf)
print(svc_acc) # 0.6562
```

    0.6562


## Tree Baseed Models

### Decision Trees


```python
from sklearn.tree import DecisionTreeClassifier as DTC
dtc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',DTC())])
dtc_clf.fit(X=X_train, y=y_train)
dtc_acc, dtc_predictions = imdb_acc(dtc_clf)
dtc_acc
```




    0.7068



## Random Forest Classifier 


```python
from sklearn.ensemble import RandomForestClassifier as RFC
rfc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',RFC())])
rfc_clf.fit(X=X_train, y=y_train)
rfc_acc, rfc_predictions = imdb_acc(rfc_clf)
rfc_acc
```




    0.73472



## Extra Trees Classifier 


```python
from sklearn.ensemble import ExtraTreesClassifier as XTC
xtc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',XTC())])
xtc_clf.fit(X=X_train, y=y_train)
xtc_acc, xtc_predictions = imdb_acc(xtc_clf)
xtc_acc
```




    0.75096



# Automatically Fine Tuning 

### RandomizedSearch


```python
from sklearn.model_selection import RandomizedSearchCV
```


```python
from sklearn.model_selection import RandomizedSearchCV
param_grid = dict(clf__C=[50, 75, 85, 100], 
                  vect__stop_words=['english', None],
                  vect__ngram_range = [(1, 1), (1, 3)],
                  vect__lowercase = [True, False],
                 )
```


```python
random_search = RandomizedSearchCV(lr_clf, param_distributions=param_grid, n_iter=5, scoring='accuracy', n_jobs=-1, cv=3)
random_search.fit(X_train, y_train)
```




    RandomizedSearchCV(cv=3, error_score='raise',
              estimator=Pipeline(memory=None,
         steps=[('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 3), preprocessor=None, stop_words='english',
            ...ty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False))]),
              fit_params=None, iid=True, n_iter=5, n_jobs=-1,
              param_distributions={'clf__C': [50, 75, 85, 100], 'vect__stop_words': ['english', None], 'vect__ngram_range': [(1, 1), (1, 3)], 'vect__lowercase': [True, False]},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              return_train_score='warn', scoring='accuracy', verbose=0)




```python
print(f'Calculated cross-validation accuracy: {random_search.best_score_}')
```

    Calculated cross-validation accuracy: 0.89884



```python
best_random_clf = random_search.best_estimator_
```


```python
best_random_clf.fit(X_train, y_train)
```




    Pipeline(memory=None,
         steps=[('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 3), preprocessor=None, stop_words=None,
            strip...ty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False))])




```python
imdb_acc(best_random_clf)

```




    (0.90096, array([1, 1, 0, ..., 0, 1, 1], dtype=int64))




```python
best_random_clf.steps
```




    [('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
              dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
              lowercase=True, max_df=1.0, max_features=None, min_df=1,
              ngram_range=(1, 3), preprocessor=None, stop_words=None,
              strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
              tokenizer=None, vocabulary=None)),
     ('tfidf',
      TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)),
     ('clf',
      LogisticRegression(C=75, class_weight=None, dual=False, fit_intercept=True,
                intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                verbose=0, warm_start=False))]




```python
lr_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))), ('tfidf', TfidfTransformer()), ('clf',LR())])
```


```python
from sklearn.model_selection import GridSearchCV
param_grid = dict(clf__C=[85, 100, 125, 150])
grid_search = GridSearchCV(lr_clf, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=3)
```


```python
%%time
grid_search.fit(X_train, y_train)
```


    ---------------------------------------------------------------------------

    RemoteTraceback                           Traceback (most recent call last)

    RemoteTraceback: 
    """
    Traceback (most recent call last):
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\_parallel_backends.py", line 350, in __call__
        return self.func(*args, **kwargs)
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py", line 131, in __call__
        return [func(*args, **kwargs) for func, args, kwargs in self.items]
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py", line 131, in <listcomp>
        return [func(*args, **kwargs) for func, args, kwargs in self.items]
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\model_selection\_validation.py", line 458, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py", line 248, in fit
        Xt, fit_params = self._fit(X, y, **fit_params)
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py", line 213, in _fit
        **fit_params_steps[name])
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\memory.py", line 362, in __call__
        return self.func(*args, **kwargs)
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py", line 581, in _fit_transform_one
        res = transformer.fit_transform(X, y, **fit_params)
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\feature_extraction\text.py", line 875, in fit_transform
        X = self._sort_features(X, vocabulary)
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\feature_extraction\text.py", line 731, in _sort_features
        X.indices = map_index.take(X.indices, mode='clip')
    MemoryError
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "D:\Miniconda3\envs\nlp\lib\multiprocessing\pool.py", line 119, in worker
        result = (True, func(*args, **kwds))
      File "D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\_parallel_backends.py", line 359, in __call__
        raise TransportableException(text, e_type)
    sklearn.externals.joblib.my_exceptions.TransportableException: TransportableException
    ___________________________________________________________________________
    MemoryError                                        Wed Sep 26 05:47:48 2018
    PID: 4472                   Python 3.6.6: D:\Miniconda3\envs\nlp\python.exe
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py in __call__(self=<sklearn.externals.joblib.parallel.BatchedCalls object>)
        126     def __init__(self, iterator_slice):
        127         self.items = list(iterator_slice)
        128         self._size = len(self.items)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
            self.items = [(<function _fit_and_score>, (Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), 0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64, {'score': make_scorer(accuracy_score)}, array([    0,     1,     2, ..., 16685, 16686, 16687]), array([16641, 16643, 16644, ..., 24997, 24998, 24999]), 0, {'clf__C': 85}), {'error_score': 'raise', 'fit_params': {}, 'return_n_test_samples': True, 'return_parameters': False, 'return_times': True, 'return_train_score': 'warn'})]
        132 
        133     def __len__(self):
        134         return self._size
        135 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py in <listcomp>(.0=<list_iterator object>)
        126     def __init__(self, iterator_slice):
        127         self.items = list(iterator_slice)
        128         self._size = len(self.items)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
            func = <function _fit_and_score>
            args = (Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), 0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64, {'score': make_scorer(accuracy_score)}, array([    0,     1,     2, ..., 16685, 16686, 16687]), array([16641, 16643, 16644, ..., 24997, 24998, 24999]), 0, {'clf__C': 85})
            kwargs = {'error_score': 'raise', 'fit_params': {}, 'return_n_test_samples': True, 'return_parameters': False, 'return_times': True, 'return_train_score': 'warn'}
        132 
        133     def __len__(self):
        134         return self._size
        135 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\model_selection\_validation.py in _fit_and_score(estimator=Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), X=0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64, scorer={'score': make_scorer(accuracy_score)}, train=array([    0,     1,     2, ..., 16685, 16686, 16687]), test=array([16641, 16643, 16644, ..., 24997, 24998, 24999]), verbose=0, parameters={'clf__C': 85}, fit_params={}, return_train_score='warn', return_parameters=False, return_n_test_samples=True, return_times=True, error_score='raise')
        453 
        454     try:
        455         if y_train is None:
        456             estimator.fit(X_train, **fit_params)
        457         else:
    --> 458             estimator.fit(X_train, y_train, **fit_params)
            estimator.fit = <bound method Pipeline.fit of Pipeline(memory=No....0001,
              verbose=0, warm_start=False))])>
            X_train = 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object
            y_train = 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64
            fit_params = {}
        459 
        460     except Exception as e:
        461         # Note fit time as time until error
        462         fit_time = time.time() - start_time
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py in fit(self=Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), X=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64, **fit_params={})
        243         Returns
        244         -------
        245         self : Pipeline
        246             This estimator
        247         """
    --> 248         Xt, fit_params = self._fit(X, y, **fit_params)
            Xt = undefined
            fit_params = {}
            self._fit = <bound method Pipeline._fit of Pipeline(memory=N....0001,
              verbose=0, warm_start=False))])>
            X = 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object
            y = 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64
        249         if self._final_estimator is not None:
        250             self._final_estimator.fit(Xt, y, **fit_params)
        251         return self
        252 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py in _fit(self=Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), X=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64, **fit_params={})
        208                 else:
        209                     cloned_transformer = clone(transformer)
        210                 # Fit or load from cache the current transfomer
        211                 Xt, fitted_transformer = fit_transform_one_cached(
        212                     cloned_transformer, None, Xt, y,
    --> 213                     **fit_params_steps[name])
            fit_params_steps = {'clf': {}, 'tfidf': {}, 'vect': {}}
            name = 'vect'
        214                 # Replace the transformer of the step with the fitted
        215                 # transformer. This is necessary when loading the transformer
        216                 # from the cache.
        217                 self.steps[step_idx] = (name, fitted_transformer)
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\memory.py in __call__(self=NotMemorizedFunc(func=<function _fit_transform_one at 0x000001D3E9EC6158>), *args=(CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), None, 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64), **kwargs={})
        357     # Should be a light as possible (for speed)
        358     def __init__(self, func):
        359         self.func = func
        360 
        361     def __call__(self, *args, **kwargs):
    --> 362         return self.func(*args, **kwargs)
            self.func = <function _fit_transform_one>
            args = (CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), None, 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64)
            kwargs = {}
        363 
        364     def call_and_shelve(self, *args, **kwargs):
        365         return NotMemorizedResult(self.func(*args, **kwargs))
        366 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py in _fit_transform_one(transformer=CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), weight=None, X=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64, **fit_params={})
        576 
        577 
        578 def _fit_transform_one(transformer, weight, X, y,
        579                        **fit_params):
        580     if hasattr(transformer, 'fit_transform'):
    --> 581         res = transformer.fit_transform(X, y, **fit_params)
            res = undefined
            transformer.fit_transform = <bound method CountVectorizer.fit_transform of C...w+\\b',
            tokenizer=None, vocabulary=None)>
            X = 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object
            y = 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64
            fit_params = {}
        582     else:
        583         res = transformer.fit(X, y, **fit_params).transform(X)
        584     # if we have a weight for this transformer, multiply output
        585     if weight is None:
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\feature_extraction\text.py in fit_transform(self=CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), raw_documents=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64)
        870 
        871         if self.binary:
        872             X.data.fill(1)
        873 
        874         if not self.fixed_vocabulary_:
    --> 875             X = self._sort_features(X, vocabulary)
            X = <16668x3696114 sparse matrix of type '<class 'nu... stored elements in Compressed Sparse Row format>
            self._sort_features = <bound method CountVectorizer._sort_features of ...w+\\b',
            tokenizer=None, vocabulary=None)>
            vocabulary = <class 'dict'> instance
        876 
        877             n_doc = X.shape[0]
        878             max_doc_count = (max_df
        879                              if isinstance(max_df, numbers.Integral)
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\feature_extraction\text.py in _sort_features(self=CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), X=<16668x3696114 sparse matrix of type '<class 'nu... stored elements in Compressed Sparse Row format>, vocabulary=<class 'dict'> instance)
        726         map_index = np.empty(len(sorted_features), dtype=np.int32)
        727         for new_val, (term, old_val) in enumerate(sorted_features):
        728             vocabulary[term] = new_val
        729             map_index[old_val] = new_val
        730 
    --> 731         X.indices = map_index.take(X.indices, mode='clip')
            X.indices = array([      0,       1,       2, ..., 3696111, 3696112, 3696113],
          dtype=int32)
            map_index.take = <built-in method take of numpy.ndarray object>
        732         return X
        733 
        734     def _limit_features(self, X, vocabulary, high=None, low=None,
        735                         limit=None):
    
    MemoryError: 
    ___________________________________________________________________________
    """

    
    The above exception was the direct cause of the following exception:


    TransportableException                    Traceback (most recent call last)

    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py in retrieve(self)
        698                 if getattr(self._backend, 'supports_timeout', False):
    --> 699                     self._output.extend(job.get(timeout=self.timeout))
        700                 else:


    D:\Miniconda3\envs\nlp\lib\multiprocessing\pool.py in get(self, timeout)
        643         else:
    --> 644             raise self._value
        645 


    TransportableException: TransportableException
    ___________________________________________________________________________
    MemoryError                                        Wed Sep 26 05:47:48 2018
    PID: 4472                   Python 3.6.6: D:\Miniconda3\envs\nlp\python.exe
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py in __call__(self=<sklearn.externals.joblib.parallel.BatchedCalls object>)
        126     def __init__(self, iterator_slice):
        127         self.items = list(iterator_slice)
        128         self._size = len(self.items)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
            self.items = [(<function _fit_and_score>, (Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), 0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64, {'score': make_scorer(accuracy_score)}, array([    0,     1,     2, ..., 16685, 16686, 16687]), array([16641, 16643, 16644, ..., 24997, 24998, 24999]), 0, {'clf__C': 85}), {'error_score': 'raise', 'fit_params': {}, 'return_n_test_samples': True, 'return_parameters': False, 'return_times': True, 'return_train_score': 'warn'})]
        132 
        133     def __len__(self):
        134         return self._size
        135 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py in <listcomp>(.0=<list_iterator object>)
        126     def __init__(self, iterator_slice):
        127         self.items = list(iterator_slice)
        128         self._size = len(self.items)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
            func = <function _fit_and_score>
            args = (Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), 0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64, {'score': make_scorer(accuracy_score)}, array([    0,     1,     2, ..., 16685, 16686, 16687]), array([16641, 16643, 16644, ..., 24997, 24998, 24999]), 0, {'clf__C': 85})
            kwargs = {'error_score': 'raise', 'fit_params': {}, 'return_n_test_samples': True, 'return_parameters': False, 'return_times': True, 'return_train_score': 'warn'}
        132 
        133     def __len__(self):
        134         return self._size
        135 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\model_selection\_validation.py in _fit_and_score(estimator=Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), X=0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64, scorer={'score': make_scorer(accuracy_score)}, train=array([    0,     1,     2, ..., 16685, 16686, 16687]), test=array([16641, 16643, 16644, ..., 24997, 24998, 24999]), verbose=0, parameters={'clf__C': 85}, fit_params={}, return_train_score='warn', return_parameters=False, return_n_test_samples=True, return_times=True, error_score='raise')
        453 
        454     try:
        455         if y_train is None:
        456             estimator.fit(X_train, **fit_params)
        457         else:
    --> 458             estimator.fit(X_train, y_train, **fit_params)
            estimator.fit = <bound method Pipeline.fit of Pipeline(memory=No....0001,
              verbose=0, warm_start=False))])>
            X_train = 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object
            y_train = 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64
            fit_params = {}
        459 
        460     except Exception as e:
        461         # Note fit time as time until error
        462         fit_time = time.time() - start_time
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py in fit(self=Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), X=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64, **fit_params={})
        243         Returns
        244         -------
        245         self : Pipeline
        246             This estimator
        247         """
    --> 248         Xt, fit_params = self._fit(X, y, **fit_params)
            Xt = undefined
            fit_params = {}
            self._fit = <bound method Pipeline._fit of Pipeline(memory=N....0001,
              verbose=0, warm_start=False))])>
            X = 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object
            y = 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64
        249         if self._final_estimator is not None:
        250             self._final_estimator.fit(Xt, y, **fit_params)
        251         return self
        252 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py in _fit(self=Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), X=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64, **fit_params={})
        208                 else:
        209                     cloned_transformer = clone(transformer)
        210                 # Fit or load from cache the current transfomer
        211                 Xt, fitted_transformer = fit_transform_one_cached(
        212                     cloned_transformer, None, Xt, y,
    --> 213                     **fit_params_steps[name])
            fit_params_steps = {'clf': {}, 'tfidf': {}, 'vect': {}}
            name = 'vect'
        214                 # Replace the transformer of the step with the fitted
        215                 # transformer. This is necessary when loading the transformer
        216                 # from the cache.
        217                 self.steps[step_idx] = (name, fitted_transformer)
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\memory.py in __call__(self=NotMemorizedFunc(func=<function _fit_transform_one at 0x000001D3E9EC6158>), *args=(CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), None, 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64), **kwargs={})
        357     # Should be a light as possible (for speed)
        358     def __init__(self, func):
        359         self.func = func
        360 
        361     def __call__(self, *args, **kwargs):
    --> 362         return self.func(*args, **kwargs)
            self.func = <function _fit_transform_one>
            args = (CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), None, 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64)
            kwargs = {}
        363 
        364     def call_and_shelve(self, *args, **kwargs):
        365         return NotMemorizedResult(self.func(*args, **kwargs))
        366 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py in _fit_transform_one(transformer=CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), weight=None, X=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64, **fit_params={})
        576 
        577 
        578 def _fit_transform_one(transformer, weight, X, y,
        579                        **fit_params):
        580     if hasattr(transformer, 'fit_transform'):
    --> 581         res = transformer.fit_transform(X, y, **fit_params)
            res = undefined
            transformer.fit_transform = <bound method CountVectorizer.fit_transform of C...w+\\b',
            tokenizer=None, vocabulary=None)>
            X = 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object
            y = 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64
            fit_params = {}
        582     else:
        583         res = transformer.fit(X, y, **fit_params).transform(X)
        584     # if we have a weight for this transformer, multiply output
        585     if weight is None:
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\feature_extraction\text.py in fit_transform(self=CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), raw_documents=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64)
        870 
        871         if self.binary:
        872             X.data.fill(1)
        873 
        874         if not self.fixed_vocabulary_:
    --> 875             X = self._sort_features(X, vocabulary)
            X = <16668x3696114 sparse matrix of type '<class 'nu... stored elements in Compressed Sparse Row format>
            self._sort_features = <bound method CountVectorizer._sort_features of ...w+\\b',
            tokenizer=None, vocabulary=None)>
            vocabulary = <class 'dict'> instance
        876 
        877             n_doc = X.shape[0]
        878             max_doc_count = (max_df
        879                              if isinstance(max_df, numbers.Integral)
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\feature_extraction\text.py in _sort_features(self=CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), X=<16668x3696114 sparse matrix of type '<class 'nu... stored elements in Compressed Sparse Row format>, vocabulary=<class 'dict'> instance)
        726         map_index = np.empty(len(sorted_features), dtype=np.int32)
        727         for new_val, (term, old_val) in enumerate(sorted_features):
        728             vocabulary[term] = new_val
        729             map_index[old_val] = new_val
        730 
    --> 731         X.indices = map_index.take(X.indices, mode='clip')
            X.indices = array([      0,       1,       2, ..., 3696111, 3696112, 3696113],
          dtype=int32)
            map_index.take = <built-in method take of numpy.ndarray object>
        732         return X
        733 
        734     def _limit_features(self, X, vocabulary, high=None, low=None,
        735                         limit=None):
    
    MemoryError: 
    ___________________________________________________________________________

    
    During handling of the above exception, another exception occurred:


    JoblibMemoryError                         Traceback (most recent call last)

    <timed eval> in <module>()


    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\model_selection\_search.py in fit(self, X, y, groups, **fit_params)
        638                                   error_score=self.error_score)
        639           for parameters, (train, test) in product(candidate_params,
    --> 640                                                    cv.split(X, y, groups)))
        641 
        642         # if one choose to see train score, "out" will contain train score info


    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py in __call__(self, iterable)
        787                 # consumption.
        788                 self._iterating = False
    --> 789             self.retrieve()
        790             # Make sure that we get a last message telling us we are done
        791             elapsed_time = time.time() - self._start_time


    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py in retrieve(self)
        738                     exception = exception_type(report)
        739 
    --> 740                     raise exception
        741 
        742     def __call__(self, iterable):


    JoblibMemoryError: JoblibMemoryError
    ___________________________________________________________________________
    Multiprocessing exception:
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\runpy.py in _run_module_as_main(mod_name='ipykernel.__main__', alter_argv=1)
        188         sys.exit(msg)
        189     main_globals = sys.modules["__main__"].__dict__
        190     if alter_argv:
        191         sys.argv[0] = mod_spec.origin
        192     return _run_code(code, main_globals, None,
    --> 193                      "__main__", mod_spec)
            mod_spec = ModuleSpec(name='ipykernel.__main__', loader=<_f...nlp\\lib\\site-packages\\ipykernel\\__main__.py')
        194 
        195 def run_module(mod_name, init_globals=None,
        196                run_name=None, alter_sys=False):
        197     """Execute a module's code without importing it
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\runpy.py in _run_code(code=<code object <module> at 0x000001B7102B10C0, fil...lib\site-packages\ipykernel\__main__.py", line 1>, run_globals={'__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__cached__': r'D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\__pycache__\__main__.cpython-36.pyc', '__doc__': None, '__file__': r'D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\__main__.py', '__loader__': <_frozen_importlib_external.SourceFileLoader object>, '__name__': '__main__', '__package__': 'ipykernel', '__spec__': ModuleSpec(name='ipykernel.__main__', loader=<_f...nlp\\lib\\site-packages\\ipykernel\\__main__.py'), 'app': <module 'ipykernel.kernelapp' from 'D:\\Minicond...lp\\lib\\site-packages\\ipykernel\\kernelapp.py'>}, init_globals=None, mod_name='__main__', mod_spec=ModuleSpec(name='ipykernel.__main__', loader=<_f...nlp\\lib\\site-packages\\ipykernel\\__main__.py'), pkg_name='ipykernel', script_name=None)
         80                        __cached__ = cached,
         81                        __doc__ = None,
         82                        __loader__ = loader,
         83                        __package__ = pkg_name,
         84                        __spec__ = mod_spec)
    ---> 85     exec(code, run_globals)
            code = <code object <module> at 0x000001B7102B10C0, fil...lib\site-packages\ipykernel\__main__.py", line 1>
            run_globals = {'__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__cached__': r'D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\__pycache__\__main__.cpython-36.pyc', '__doc__': None, '__file__': r'D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\__main__.py', '__loader__': <_frozen_importlib_external.SourceFileLoader object>, '__name__': '__main__', '__package__': 'ipykernel', '__spec__': ModuleSpec(name='ipykernel.__main__', loader=<_f...nlp\\lib\\site-packages\\ipykernel\\__main__.py'), 'app': <module 'ipykernel.kernelapp' from 'D:\\Minicond...lp\\lib\\site-packages\\ipykernel\\kernelapp.py'>}
         86     return run_globals
         87 
         88 def _run_module_code(code, init_globals=None,
         89                     mod_name=None, mod_spec=None,
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\__main__.py in <module>()
          1 if __name__ == '__main__':
          2     from ipykernel import kernelapp as app
    ----> 3     app.launch_new_instance()
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\traitlets\config\application.py in launch_instance(cls=<class 'ipykernel.kernelapp.IPKernelApp'>, argv=None, **kwargs={})
        653 
        654         If a global instance already exists, this reinitializes and starts it
        655         """
        656         app = cls.instance(**kwargs)
        657         app.initialize(argv)
    --> 658         app.start()
            app.start = <bound method IPKernelApp.start of <ipykernel.kernelapp.IPKernelApp object>>
        659 
        660 #-----------------------------------------------------------------------------
        661 # utility functions, for convenience
        662 #-----------------------------------------------------------------------------
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\kernelapp.py in start(self=<ipykernel.kernelapp.IPKernelApp object>)
        481         if self.poller is not None:
        482             self.poller.start()
        483         self.kernel.start()
        484         self.io_loop = ioloop.IOLoop.current()
        485         try:
    --> 486             self.io_loop.start()
            self.io_loop.start = <bound method BaseAsyncIOLoop.start of <tornado.platform.asyncio.AsyncIOMainLoop object>>
        487         except KeyboardInterrupt:
        488             pass
        489 
        490 launch_new_instance = IPKernelApp.launch_instance
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\tornado\platform\asyncio.py in start(self=<tornado.platform.asyncio.AsyncIOMainLoop object>)
        122         except (RuntimeError, AssertionError):
        123             old_loop = None
        124         try:
        125             self._setup_logging()
        126             asyncio.set_event_loop(self.asyncio_loop)
    --> 127             self.asyncio_loop.run_forever()
            self.asyncio_loop.run_forever = <bound method BaseEventLoop.run_forever of <_Win...EventLoop running=True closed=False debug=False>>
        128         finally:
        129             asyncio.set_event_loop(old_loop)
        130 
        131     def stop(self):
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\asyncio\base_events.py in run_forever(self=<_WindowsSelectorEventLoop running=True closed=False debug=False>)
        417             sys.set_asyncgen_hooks(firstiter=self._asyncgen_firstiter_hook,
        418                                    finalizer=self._asyncgen_finalizer_hook)
        419         try:
        420             events._set_running_loop(self)
        421             while True:
    --> 422                 self._run_once()
            self._run_once = <bound method BaseEventLoop._run_once of <_Windo...EventLoop running=True closed=False debug=False>>
        423                 if self._stopping:
        424                     break
        425         finally:
        426             self._stopping = False
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\asyncio\base_events.py in _run_once(self=<_WindowsSelectorEventLoop running=True closed=False debug=False>)
       1429                         logger.warning('Executing %s took %.3f seconds',
       1430                                        _format_handle(handle), dt)
       1431                 finally:
       1432                     self._current_handle = None
       1433             else:
    -> 1434                 handle._run()
            handle._run = <bound method Handle._run of <Handle IOLoop._run_callback(functools.par...01B7150CA268>))>>
       1435         handle = None  # Needed to break cycles when an exception occurs.
       1436 
       1437     def _set_coroutine_wrapper(self, enabled):
       1438         try:
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\asyncio\events.py in _run(self=<Handle IOLoop._run_callback(functools.par...01B7150CA268>))>)
        140             self._callback = None
        141             self._args = None
        142 
        143     def _run(self):
        144         try:
    --> 145             self._callback(*self._args)
            self._callback = <bound method IOLoop._run_callback of <tornado.platform.asyncio.AsyncIOMainLoop object>>
            self._args = (functools.partial(<function wrap.<locals>.null_wrapper at 0x000001B7150CA268>),)
        146         except Exception as exc:
        147             cb = _format_callback_source(self._callback, self._args)
        148             msg = 'Exception in callback {}'.format(cb)
        149             context = {
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\tornado\ioloop.py in _run_callback(self=<tornado.platform.asyncio.AsyncIOMainLoop object>, callback=functools.partial(<function wrap.<locals>.null_wrapper at 0x000001B7150CA268>))
        754         """Runs a callback with error handling.
        755 
        756         For use in subclasses.
        757         """
        758         try:
    --> 759             ret = callback()
            ret = undefined
            callback = functools.partial(<function wrap.<locals>.null_wrapper at 0x000001B7150CA268>)
        760             if ret is not None:
        761                 from tornado import gen
        762                 # Functions that return Futures typically swallow all
        763                 # exceptions and store them in the Future.  If a Future
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\tornado\stack_context.py in null_wrapper(*args=(), **kwargs={})
        271         # Fast path when there are no active contexts.
        272         def null_wrapper(*args, **kwargs):
        273             try:
        274                 current_state = _state.contexts
        275                 _state.contexts = cap_contexts[0]
    --> 276                 return fn(*args, **kwargs)
            args = ()
            kwargs = {}
        277             finally:
        278                 _state.contexts = current_state
        279         null_wrapper._wrapped = True
        280         return null_wrapper
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\zmq\eventloop\zmqstream.py in <lambda>()
        531             return
        532 
        533         if state & self.socket.events:
        534             # events still exist that haven't been processed
        535             # explicitly schedule handling to avoid missing events due to edge-triggered FDs
    --> 536             self.io_loop.add_callback(lambda : self._handle_events(self.socket, 0))
        537 
        538     def _init_io_state(self):
        539         """initialize the ioloop event handler"""
        540         with stack_context.NullContext():
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\zmq\eventloop\zmqstream.py in _handle_events(self=<zmq.eventloop.zmqstream.ZMQStream object>, fd=<zmq.sugar.socket.Socket object>, events=0)
        445             return
        446         zmq_events = self.socket.EVENTS
        447         try:
        448             # dispatch events:
        449             if zmq_events & zmq.POLLIN and self.receiving():
    --> 450                 self._handle_recv()
            self._handle_recv = <bound method ZMQStream._handle_recv of <zmq.eventloop.zmqstream.ZMQStream object>>
        451                 if not self.socket:
        452                     return
        453             if zmq_events & zmq.POLLOUT and self.sending():
        454                 self._handle_send()
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\zmq\eventloop\zmqstream.py in _handle_recv(self=<zmq.eventloop.zmqstream.ZMQStream object>)
        475             else:
        476                 raise
        477         else:
        478             if self._recv_callback:
        479                 callback = self._recv_callback
    --> 480                 self._run_callback(callback, msg)
            self._run_callback = <bound method ZMQStream._run_callback of <zmq.eventloop.zmqstream.ZMQStream object>>
            callback = <function wrap.<locals>.null_wrapper>
            msg = [<zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>]
        481         
        482 
        483     def _handle_send(self):
        484         """Handle a send event."""
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\zmq\eventloop\zmqstream.py in _run_callback(self=<zmq.eventloop.zmqstream.ZMQStream object>, callback=<function wrap.<locals>.null_wrapper>, *args=([<zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>],), **kwargs={})
        427         close our socket."""
        428         try:
        429             # Use a NullContext to ensure that all StackContexts are run
        430             # inside our blanket exception handler rather than outside.
        431             with stack_context.NullContext():
    --> 432                 callback(*args, **kwargs)
            callback = <function wrap.<locals>.null_wrapper>
            args = ([<zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>],)
            kwargs = {}
        433         except:
        434             gen_log.error("Uncaught exception in ZMQStream callback",
        435                           exc_info=True)
        436             # Re-raise the exception so that IOLoop.handle_callback_exception
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\tornado\stack_context.py in null_wrapper(*args=([<zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>],), **kwargs={})
        271         # Fast path when there are no active contexts.
        272         def null_wrapper(*args, **kwargs):
        273             try:
        274                 current_state = _state.contexts
        275                 _state.contexts = cap_contexts[0]
    --> 276                 return fn(*args, **kwargs)
            args = ([<zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>],)
            kwargs = {}
        277             finally:
        278                 _state.contexts = current_state
        279         null_wrapper._wrapped = True
        280         return null_wrapper
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\kernelbase.py in dispatcher(msg=[<zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>])
        278         if self.control_stream:
        279             self.control_stream.on_recv(self.dispatch_control, copy=False)
        280 
        281         def make_dispatcher(stream):
        282             def dispatcher(msg):
    --> 283                 return self.dispatch_shell(stream, msg)
            msg = [<zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>, <zmq.sugar.frame.Frame object>]
        284             return dispatcher
        285 
        286         for s in self.shell_streams:
        287             s.on_recv(make_dispatcher(s), copy=False)
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\kernelbase.py in dispatch_shell(self=<ipykernel.ipkernel.IPythonKernel object>, stream=<zmq.eventloop.zmqstream.ZMQStream object>, msg={'buffers': [], 'content': {'allow_stdin': True, 'code': '%%time\ngrid_search.fit(X_train, y_train)', 'silent': False, 'stop_on_error': True, 'store_history': True, 'user_expressions': {}}, 'header': {'date': datetime.datetime(2018, 9, 25, 23, 40, 27, 639848, tzinfo=tzutc()), 'msg_id': '0a644ca9c93a422b8ffb37b847a55c64', 'msg_type': 'execute_request', 'session': 'a6f2e7c3cb634e10b16d91097af28c58', 'username': 'username', 'version': '5.2'}, 'metadata': {}, 'msg_id': '0a644ca9c93a422b8ffb37b847a55c64', 'msg_type': 'execute_request', 'parent_header': {}})
        228             self.log.warn("Unknown message type: %r", msg_type)
        229         else:
        230             self.log.debug("%s: %s", msg_type, msg)
        231             self.pre_handler_hook()
        232             try:
    --> 233                 handler(stream, idents, msg)
            handler = <bound method Kernel.execute_request of <ipykernel.ipkernel.IPythonKernel object>>
            stream = <zmq.eventloop.zmqstream.ZMQStream object>
            idents = [b'a6f2e7c3cb634e10b16d91097af28c58']
            msg = {'buffers': [], 'content': {'allow_stdin': True, 'code': '%%time\ngrid_search.fit(X_train, y_train)', 'silent': False, 'stop_on_error': True, 'store_history': True, 'user_expressions': {}}, 'header': {'date': datetime.datetime(2018, 9, 25, 23, 40, 27, 639848, tzinfo=tzutc()), 'msg_id': '0a644ca9c93a422b8ffb37b847a55c64', 'msg_type': 'execute_request', 'session': 'a6f2e7c3cb634e10b16d91097af28c58', 'username': 'username', 'version': '5.2'}, 'metadata': {}, 'msg_id': '0a644ca9c93a422b8ffb37b847a55c64', 'msg_type': 'execute_request', 'parent_header': {}}
        234             except Exception:
        235                 self.log.error("Exception in message handler:", exc_info=True)
        236             finally:
        237                 self.post_handler_hook()
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\kernelbase.py in execute_request(self=<ipykernel.ipkernel.IPythonKernel object>, stream=<zmq.eventloop.zmqstream.ZMQStream object>, ident=[b'a6f2e7c3cb634e10b16d91097af28c58'], parent={'buffers': [], 'content': {'allow_stdin': True, 'code': '%%time\ngrid_search.fit(X_train, y_train)', 'silent': False, 'stop_on_error': True, 'store_history': True, 'user_expressions': {}}, 'header': {'date': datetime.datetime(2018, 9, 25, 23, 40, 27, 639848, tzinfo=tzutc()), 'msg_id': '0a644ca9c93a422b8ffb37b847a55c64', 'msg_type': 'execute_request', 'session': 'a6f2e7c3cb634e10b16d91097af28c58', 'username': 'username', 'version': '5.2'}, 'metadata': {}, 'msg_id': '0a644ca9c93a422b8ffb37b847a55c64', 'msg_type': 'execute_request', 'parent_header': {}})
        394         if not silent:
        395             self.execution_count += 1
        396             self._publish_execute_input(code, parent, self.execution_count)
        397 
        398         reply_content = self.do_execute(code, silent, store_history,
    --> 399                                         user_expressions, allow_stdin)
            user_expressions = {}
            allow_stdin = True
        400 
        401         # Flush output before sending the reply.
        402         sys.stdout.flush()
        403         sys.stderr.flush()
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\ipkernel.py in do_execute(self=<ipykernel.ipkernel.IPythonKernel object>, code='%%time\ngrid_search.fit(X_train, y_train)', silent=False, store_history=True, user_expressions={}, allow_stdin=True)
        203 
        204         self._forward_input(allow_stdin)
        205 
        206         reply_content = {}
        207         try:
    --> 208             res = shell.run_cell(code, store_history=store_history, silent=silent)
            res = undefined
            shell.run_cell = <bound method ZMQInteractiveShell.run_cell of <ipykernel.zmqshell.ZMQInteractiveShell object>>
            code = '%%time\ngrid_search.fit(X_train, y_train)'
            store_history = True
            silent = False
        209         finally:
        210             self._restore_input()
        211 
        212         if res.error_before_exec is not None:
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel\zmqshell.py in run_cell(self=<ipykernel.zmqshell.ZMQInteractiveShell object>, *args=('%%time\ngrid_search.fit(X_train, y_train)',), **kwargs={'silent': False, 'store_history': True})
        532             )
        533         self.payload_manager.write_payload(payload)
        534 
        535     def run_cell(self, *args, **kwargs):
        536         self._last_traceback = None
    --> 537         return super(ZMQInteractiveShell, self).run_cell(*args, **kwargs)
            self.run_cell = <bound method ZMQInteractiveShell.run_cell of <ipykernel.zmqshell.ZMQInteractiveShell object>>
            args = ('%%time\ngrid_search.fit(X_train, y_train)',)
            kwargs = {'silent': False, 'store_history': True}
        538 
        539     def _showtraceback(self, etype, evalue, stb):
        540         # try to preserve ordering of tracebacks and print statements
        541         sys.stdout.flush()
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\IPython\core\interactiveshell.py in run_cell(self=<ipykernel.zmqshell.ZMQInteractiveShell object>, raw_cell='%%time\ngrid_search.fit(X_train, y_train)', store_history=True, silent=False, shell_futures=True)
       2657         -------
       2658         result : :class:`ExecutionResult`
       2659         """
       2660         try:
       2661             result = self._run_cell(
    -> 2662                 raw_cell, store_history, silent, shell_futures)
            raw_cell = '%%time\ngrid_search.fit(X_train, y_train)'
            store_history = True
            silent = False
            shell_futures = True
       2663         finally:
       2664             self.events.trigger('post_execute')
       2665             if not silent:
       2666                 self.events.trigger('post_run_cell', result)
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\IPython\core\interactiveshell.py in _run_cell(self=<ipykernel.zmqshell.ZMQInteractiveShell object>, raw_cell='%%time\ngrid_search.fit(X_train, y_train)', store_history=True, silent=False, shell_futures=True)
       2780                 self.displayhook.exec_result = result
       2781 
       2782                 # Execute the user code
       2783                 interactivity = 'none' if silent else self.ast_node_interactivity
       2784                 has_raised = self.run_ast_nodes(code_ast.body, cell_name,
    -> 2785                    interactivity=interactivity, compiler=compiler, result=result)
            interactivity = 'last_expr'
            compiler = <IPython.core.compilerop.CachingCompiler object>
       2786                 
       2787                 self.last_execution_succeeded = not has_raised
       2788                 self.last_execution_result = result
       2789 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\IPython\core\interactiveshell.py in run_ast_nodes(self=<ipykernel.zmqshell.ZMQInteractiveShell object>, nodelist=[<_ast.Expr object>], cell_name='<ipython-input-44-9855dc574354>', interactivity='last', compiler=<IPython.core.compilerop.CachingCompiler object>, result=<ExecutionResult object at 1b72d7d5e48, executio...rue silent=False shell_futures=True> result=None>)
       2904                     return True
       2905 
       2906             for i, node in enumerate(to_run_interactive):
       2907                 mod = ast.Interactive([node])
       2908                 code = compiler(mod, cell_name, "single")
    -> 2909                 if self.run_code(code, result):
            self.run_code = <bound method InteractiveShell.run_code of <ipykernel.zmqshell.ZMQInteractiveShell object>>
            code = <code object <module> at 0x000001B7176F58A0, file "<ipython-input-44-9855dc574354>", line 1>
            result = <ExecutionResult object at 1b72d7d5e48, executio...rue silent=False shell_futures=True> result=None>
       2910                     return True
       2911 
       2912             # Flush softspace
       2913             if softspace(sys.stdout, 0):
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\IPython\core\interactiveshell.py in run_code(self=<ipykernel.zmqshell.ZMQInteractiveShell object>, code_obj=<code object <module> at 0x000001B7176F58A0, file "<ipython-input-44-9855dc574354>", line 1>, result=<ExecutionResult object at 1b72d7d5e48, executio...rue silent=False shell_futures=True> result=None>)
       2958         outflag = True  # happens in more places, so it's easier as default
       2959         try:
       2960             try:
       2961                 self.hooks.pre_run_code_hook()
       2962                 #rprint('Running code', repr(code_obj)) # dbg
    -> 2963                 exec(code_obj, self.user_global_ns, self.user_ns)
            code_obj = <code object <module> at 0x000001B7176F58A0, file "<ipython-input-44-9855dc574354>", line 1>
            self.user_global_ns = {'CountVectorizer': <class 'sklearn.feature_extraction.text.CountVectorizer'>, 'DTC': <class 'sklearn.tree.tree.DecisionTreeClassifier'>, 'GridSearchCV': <class 'sklearn.model_selection._search.GridSearchCV'>, 'In': ['', 'from pathlib import Path\nimport pandas as pd\nimp...the fastAI environment, all of these imports work', 'class TqdmUpTo(tqdm):\n    def update_to(self, b=...l = tsize\n        self.update(b * bsize - self.n)', 'def get_data(url, filename):\n    """\n    Downloa...rlretrieve(url, filename, reporthook=t.update_to)', "# Let's download some data:\ndata_url = 'http://f...clImdb.tgz'\n# get_data(data_url, 'data/imdb.tgz')", "data_path = Path(os.getcwd())/'data'/'aclImdb'\nassert data_path.exists()", 'for pathroute in os.walk(data_path):\n    next_pa...1]\n    for stop in next_path:\n        print(stop)', "train_path = data_path/'train'\ntest_path = data_path/'test'", 'def read_data(dir_path):\n    """read data into p...ce=True) # drop the column \'index\' \n    return df', "train_path = data_path/'train'\ntest_path = data_path/'test'", r"get_ipython().run_cell_magic('time', '', 'train ...d_data(train_path)\ntest = read_data(test_path)')", 'test[:5]', "# test.to_csv(data_path/'test.csv', index=False)", "# train.to_csv(data_path/'train.csv', index=False)", "X_train, y_train = train['text'], train['label']\nX_test, y_test = test['text'], test['label']", 'from sklearn.pipeline import Pipeline\nfrom sklea...ion.text import CountVectorizer, TfidfTransformer', 'from sklearn.linear_model import LogisticRegression as LR', "lr_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',LR())])", "get_ipython().run_cell_magic('time', '', 'lr_clf...re inplace, and the Pipeline is not re-assigned')", 'lr_predicted = lr_clf.predict(X_test)', ...], 'LR': <class 'sklearn.linear_model.logistic.LogisticRegression'>, 'MNB': <class 'sklearn.naive_bayes.MultinomialNB'>, 'Out': {11:                                                 ... one of my favorite movies ever! along ...      1, 18: Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), 20: 0.88312, 22: 0.879, 23: 0.86596, 25: 0.81356, 26: 0.82956, 27: 0.82992, 28: 0.8572, 29: 0.8572, ...}, 'Path': <class 'pathlib.Path'>, 'Pipeline': <class 'sklearn.pipeline.Pipeline'>, 'RFC': <class 'sklearn.ensemble.forest.RandomForestClassifier'>, ...}
            self.user_ns = {'CountVectorizer': <class 'sklearn.feature_extraction.text.CountVectorizer'>, 'DTC': <class 'sklearn.tree.tree.DecisionTreeClassifier'>, 'GridSearchCV': <class 'sklearn.model_selection._search.GridSearchCV'>, 'In': ['', 'from pathlib import Path\nimport pandas as pd\nimp...the fastAI environment, all of these imports work', 'class TqdmUpTo(tqdm):\n    def update_to(self, b=...l = tsize\n        self.update(b * bsize - self.n)', 'def get_data(url, filename):\n    """\n    Downloa...rlretrieve(url, filename, reporthook=t.update_to)', "# Let's download some data:\ndata_url = 'http://f...clImdb.tgz'\n# get_data(data_url, 'data/imdb.tgz')", "data_path = Path(os.getcwd())/'data'/'aclImdb'\nassert data_path.exists()", 'for pathroute in os.walk(data_path):\n    next_pa...1]\n    for stop in next_path:\n        print(stop)', "train_path = data_path/'train'\ntest_path = data_path/'test'", 'def read_data(dir_path):\n    """read data into p...ce=True) # drop the column \'index\' \n    return df', "train_path = data_path/'train'\ntest_path = data_path/'test'", r"get_ipython().run_cell_magic('time', '', 'train ...d_data(train_path)\ntest = read_data(test_path)')", 'test[:5]', "# test.to_csv(data_path/'test.csv', index=False)", "# train.to_csv(data_path/'train.csv', index=False)", "X_train, y_train = train['text'], train['label']\nX_test, y_test = test['text'], test['label']", 'from sklearn.pipeline import Pipeline\nfrom sklea...ion.text import CountVectorizer, TfidfTransformer', 'from sklearn.linear_model import LogisticRegression as LR', "lr_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',LR())])", "get_ipython().run_cell_magic('time', '', 'lr_clf...re inplace, and the Pipeline is not re-assigned')", 'lr_predicted = lr_clf.predict(X_test)', ...], 'LR': <class 'sklearn.linear_model.logistic.LogisticRegression'>, 'MNB': <class 'sklearn.naive_bayes.MultinomialNB'>, 'Out': {11:                                                 ... one of my favorite movies ever! along ...      1, 18: Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), 20: 0.88312, 22: 0.879, 23: 0.86596, 25: 0.81356, 26: 0.82956, 27: 0.82992, 28: 0.8572, 29: 0.8572, ...}, 'Path': <class 'pathlib.Path'>, 'Pipeline': <class 'sklearn.pipeline.Pipeline'>, 'RFC': <class 'sklearn.ensemble.forest.RandomForestClassifier'>, ...}
       2964             finally:
       2965                 # Reset our crash handler in place
       2966                 sys.excepthook = old_excepthook
       2967         except SystemExit as e:
    
    ...........................................................................
    C:\Users\nirantk\Desktop\nlp-python-deep-learning\<ipython-input-44-9855dc574354> in <module>()
    ----> 1 get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)')
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\IPython\core\interactiveshell.py in run_cell_magic(self=<ipykernel.zmqshell.ZMQInteractiveShell object>, magic_name='time', line='', cell='grid_search.fit(X_train, y_train)')
       2162             # This will need to be updated if the internal calling logic gets
       2163             # refactored, or else we'll be expanding the wrong variables.
       2164             stack_depth = 2
       2165             magic_arg_s = self.var_expand(line, stack_depth)
       2166             with self.builtin_trap:
    -> 2167                 result = fn(magic_arg_s, cell)
            result = undefined
            fn = <bound method ExecutionMagics.time of <IPython.core.magics.execution.ExecutionMagics object>>
            magic_arg_s = ''
            cell = 'grid_search.fit(X_train, y_train)'
       2168             return result
       2169 
       2170     def find_line_magic(self, magic_name):
       2171         """Find and return a line magic by name.
    
    ...........................................................................
    C:\Users\nirantk\Desktop\nlp-python-deep-learning\<decorator-gen-63> in time(self=<IPython.core.magics.execution.ExecutionMagics object>, line='', cell='grid_search.fit(X_train, y_train)', local_ns=None)
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\IPython\core\magic.py in <lambda>(f=<function ExecutionMagics.time>, *a=(<IPython.core.magics.execution.ExecutionMagics object>, '', 'grid_search.fit(X_train, y_train)', None), **k={})
        182     validate_type(magic_kind)
        183 
        184     # This is a closure to capture the magic_kind.  We could also use a class,
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
            f = <function ExecutionMagics.time>
            a = (<IPython.core.magics.execution.ExecutionMagics object>, '', 'grid_search.fit(X_train, y_train)', None)
            k = {}
        188 
        189         if callable(arg):
        190             # "Naked" decorator call (just @foo, no args)
        191             func = arg
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\IPython\core\magics\execution.py in time(self=<IPython.core.magics.execution.ExecutionMagics object>, line='', cell='grid_search.fit(X_train, y_train)', local_ns=None)
       1225         # time execution
       1226         wall_st = wtime()
       1227         if mode=='eval':
       1228             st = clock2()
       1229             try:
    -> 1230                 out = eval(code, glob, local_ns)
            out = undefined
            code = <code object <module> at 0x000001B72D7D24B0, file "<timed eval>", line 1>
            glob = {'CountVectorizer': <class 'sklearn.feature_extraction.text.CountVectorizer'>, 'DTC': <class 'sklearn.tree.tree.DecisionTreeClassifier'>, 'GridSearchCV': <class 'sklearn.model_selection._search.GridSearchCV'>, 'In': ['', 'from pathlib import Path\nimport pandas as pd\nimp...the fastAI environment, all of these imports work', 'class TqdmUpTo(tqdm):\n    def update_to(self, b=...l = tsize\n        self.update(b * bsize - self.n)', 'def get_data(url, filename):\n    """\n    Downloa...rlretrieve(url, filename, reporthook=t.update_to)', "# Let's download some data:\ndata_url = 'http://f...clImdb.tgz'\n# get_data(data_url, 'data/imdb.tgz')", "data_path = Path(os.getcwd())/'data'/'aclImdb'\nassert data_path.exists()", 'for pathroute in os.walk(data_path):\n    next_pa...1]\n    for stop in next_path:\n        print(stop)', "train_path = data_path/'train'\ntest_path = data_path/'test'", 'def read_data(dir_path):\n    """read data into p...ce=True) # drop the column \'index\' \n    return df', "train_path = data_path/'train'\ntest_path = data_path/'test'", r"get_ipython().run_cell_magic('time', '', 'train ...d_data(train_path)\ntest = read_data(test_path)')", 'test[:5]', "# test.to_csv(data_path/'test.csv', index=False)", "# train.to_csv(data_path/'train.csv', index=False)", "X_train, y_train = train['text'], train['label']\nX_test, y_test = test['text'], test['label']", 'from sklearn.pipeline import Pipeline\nfrom sklea...ion.text import CountVectorizer, TfidfTransformer', 'from sklearn.linear_model import LogisticRegression as LR', "lr_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',LR())])", "get_ipython().run_cell_magic('time', '', 'lr_clf...re inplace, and the Pipeline is not re-assigned')", 'lr_predicted = lr_clf.predict(X_test)', ...], 'LR': <class 'sklearn.linear_model.logistic.LogisticRegression'>, 'MNB': <class 'sklearn.naive_bayes.MultinomialNB'>, 'Out': {11:                                                 ... one of my favorite movies ever! along ...      1, 18: Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), 20: 0.88312, 22: 0.879, 23: 0.86596, 25: 0.81356, 26: 0.82956, 27: 0.82992, 28: 0.8572, 29: 0.8572, ...}, 'Path': <class 'pathlib.Path'>, 'Pipeline': <class 'sklearn.pipeline.Pipeline'>, 'RFC': <class 'sklearn.ensemble.forest.RandomForestClassifier'>, ...}
            local_ns = None
       1231             except:
       1232                 self.shell.showtraceback()
       1233                 return
       1234             end = clock2()
    
    ...........................................................................
    C:\Users\nirantk\Desktop\nlp-python-deep-learning\<timed eval> in <module>()
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\model_selection\_search.py in fit(self=GridSearchCV(cv=3, error_score='raise',
           e...ore='warn', scoring='accuracy',
           verbose=0), X=0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64, groups=None, **fit_params={})
        635                                   return_train_score=self.return_train_score,
        636                                   return_n_test_samples=True,
        637                                   return_times=True, return_parameters=False,
        638                                   error_score=self.error_score)
        639           for parameters, (train, test) in product(candidate_params,
    --> 640                                                    cv.split(X, y, groups)))
            cv.split = <bound method StratifiedKFold.split of Stratifie...ld(n_splits=3, random_state=None, shuffle=False)>
            X = 0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object
            y = 0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64
            groups = None
        641 
        642         # if one choose to see train score, "out" will contain train score info
        643         if self.return_train_score:
        644             (train_score_dicts, test_score_dicts, test_sample_counts, fit_time,
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py in __call__(self=Parallel(n_jobs=-1), iterable=<generator object BaseSearchCV.fit.<locals>.<genexpr>>)
        784             if pre_dispatch == "all" or n_jobs == 1:
        785                 # The iterable was consumed all at once by the above for loop.
        786                 # No need to wait for async callbacks to trigger to
        787                 # consumption.
        788                 self._iterating = False
    --> 789             self.retrieve()
            self.retrieve = <bound method Parallel.retrieve of Parallel(n_jobs=-1)>
        790             # Make sure that we get a last message telling us we are done
        791             elapsed_time = time.time() - self._start_time
        792             self._print('Done %3i out of %3i | elapsed: %s finished',
        793                         (len(self._output), len(self._output),
    
    ---------------------------------------------------------------------------
    Sub-process traceback:
    ---------------------------------------------------------------------------
    MemoryError                                        Wed Sep 26 05:47:48 2018
    PID: 4472                   Python 3.6.6: D:\Miniconda3\envs\nlp\python.exe
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py in __call__(self=<sklearn.externals.joblib.parallel.BatchedCalls object>)
        126     def __init__(self, iterator_slice):
        127         self.items = list(iterator_slice)
        128         self._size = len(self.items)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
            self.items = [(<function _fit_and_score>, (Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), 0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64, {'score': make_scorer(accuracy_score)}, array([    0,     1,     2, ..., 16685, 16686, 16687]), array([16641, 16643, 16644, ..., 24997, 24998, 24999]), 0, {'clf__C': 85}), {'error_score': 'raise', 'fit_params': {}, 'return_n_test_samples': True, 'return_parameters': False, 'return_times': True, 'return_train_score': 'warn'})]
        132 
        133     def __len__(self):
        134         return self._size
        135 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\parallel.py in <listcomp>(.0=<list_iterator object>)
        126     def __init__(self, iterator_slice):
        127         self.items = list(iterator_slice)
        128         self._size = len(self.items)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
            func = <function _fit_and_score>
            args = (Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), 0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64, {'score': make_scorer(accuracy_score)}, array([    0,     1,     2, ..., 16685, 16686, 16687]), array([16641, 16643, 16644, ..., 24997, 24998, 24999]), 0, {'clf__C': 85})
            kwargs = {'error_score': 'raise', 'fit_params': {}, 'return_n_test_samples': True, 'return_parameters': False, 'return_times': True, 'return_train_score': 'warn'}
        132 
        133     def __len__(self):
        134         return self._size
        135 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\model_selection\_validation.py in _fit_and_score(estimator=Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), X=0        Antitrust falls right into that categor...lbert...
    Name: text, Length: 25000, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...999    1
    Name: label, Length: 25000, dtype: int64, scorer={'score': make_scorer(accuracy_score)}, train=array([    0,     1,     2, ..., 16685, 16686, 16687]), test=array([16641, 16643, 16644, ..., 24997, 24998, 24999]), verbose=0, parameters={'clf__C': 85}, fit_params={}, return_train_score='warn', return_parameters=False, return_n_test_samples=True, return_times=True, error_score='raise')
        453 
        454     try:
        455         if y_train is None:
        456             estimator.fit(X_train, **fit_params)
        457         else:
    --> 458             estimator.fit(X_train, y_train, **fit_params)
            estimator.fit = <bound method Pipeline.fit of Pipeline(memory=No....0001,
              verbose=0, warm_start=False))])>
            X_train = 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object
            y_train = 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64
            fit_params = {}
        459 
        460     except Exception as e:
        461         # Note fit time as time until error
        462         fit_time = time.time() - start_time
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py in fit(self=Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), X=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64, **fit_params={})
        243         Returns
        244         -------
        245         self : Pipeline
        246             This estimator
        247         """
    --> 248         Xt, fit_params = self._fit(X, y, **fit_params)
            Xt = undefined
            fit_params = {}
            self._fit = <bound method Pipeline._fit of Pipeline(memory=N....0001,
              verbose=0, warm_start=False))])>
            X = 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object
            y = 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64
        249         if self._final_estimator is not None:
        250             self._final_estimator.fit(Xt, y, **fit_params)
        251         return self
        252 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py in _fit(self=Pipeline(memory=None,
         steps=[('vect', Count...0.0001,
              verbose=0, warm_start=False))]), X=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64, **fit_params={})
        208                 else:
        209                     cloned_transformer = clone(transformer)
        210                 # Fit or load from cache the current transfomer
        211                 Xt, fitted_transformer = fit_transform_one_cached(
        212                     cloned_transformer, None, Xt, y,
    --> 213                     **fit_params_steps[name])
            fit_params_steps = {'clf': {}, 'tfidf': {}, 'vect': {}}
            name = 'vect'
        214                 # Replace the transformer of the step with the fitted
        215                 # transformer. This is necessary when loading the transformer
        216                 # from the cache.
        217                 self.steps[step_idx] = (name, fitted_transformer)
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\externals\joblib\memory.py in __call__(self=NotMemorizedFunc(func=<function _fit_transform_one at 0x000001D3E9EC6158>), *args=(CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), None, 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64), **kwargs={})
        357     # Should be a light as possible (for speed)
        358     def __init__(self, func):
        359         self.func = func
        360 
        361     def __call__(self, *args, **kwargs):
    --> 362         return self.func(*args, **kwargs)
            self.func = <function _fit_transform_one>
            args = (CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), None, 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64)
            kwargs = {}
        363 
        364     def call_and_shelve(self, *args, **kwargs):
        365         return NotMemorizedResult(self.func(*args, **kwargs))
        366 
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\pipeline.py in _fit_transform_one(transformer=CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), weight=None, X=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64, **fit_params={})
        576 
        577 
        578 def _fit_transform_one(transformer, weight, X, y,
        579                        **fit_params):
        580     if hasattr(transformer, 'fit_transform'):
    --> 581         res = transformer.fit_transform(X, y, **fit_params)
            res = undefined
            transformer.fit_transform = <bound method CountVectorizer.fit_transform of C...w+\\b',
            tokenizer=None, vocabulary=None)>
            X = 0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object
            y = 0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64
            fit_params = {}
        582     else:
        583         res = transformer.fit(X, y, **fit_params).transform(X)
        584     # if we have a weight for this transformer, multiply output
        585     if weight is None:
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\feature_extraction\text.py in fit_transform(self=CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), raw_documents=0        Antitrust falls right into that categor...'t ac...
    Name: text, Length: 16668, dtype: object, y=0        0
    1        0
    2        1
    3        1
    4   ...687    1
    Name: label, Length: 16668, dtype: int64)
        870 
        871         if self.binary:
        872             X.data.fill(1)
        873 
        874         if not self.fixed_vocabulary_:
    --> 875             X = self._sort_features(X, vocabulary)
            X = <16668x3696114 sparse matrix of type '<class 'nu... stored elements in Compressed Sparse Row format>
            self._sort_features = <bound method CountVectorizer._sort_features of ...w+\\b',
            tokenizer=None, vocabulary=None)>
            vocabulary = <class 'dict'> instance
        876 
        877             n_doc = X.shape[0]
        878             max_doc_count = (max_df
        879                              if isinstance(max_df, numbers.Integral)
    
    ...........................................................................
    D:\Miniconda3\envs\nlp\lib\site-packages\sklearn\feature_extraction\text.py in _sort_features(self=CountVectorizer(analyzer='word', binary=False, d...\w+\\b',
            tokenizer=None, vocabulary=None), X=<16668x3696114 sparse matrix of type '<class 'nu... stored elements in Compressed Sparse Row format>, vocabulary=<class 'dict'> instance)
        726         map_index = np.empty(len(sorted_features), dtype=np.int32)
        727         for new_val, (term, old_val) in enumerate(sorted_features):
        728             vocabulary[term] = new_val
        729             map_index[old_val] = new_val
        730 
    --> 731         X.indices = map_index.take(X.indices, mode='clip')
            X.indices = array([      0,       1,       2, ..., 3696111, 3696112, 3696113],
          dtype=int32)
            map_index.take = <built-in method take of numpy.ndarray object>
        732         return X
        733 
        734     def _limit_features(self, X, vocabulary, high=None, low=None,
        735                         limit=None):
    
    MemoryError: 
    ___________________________________________________________________________



```python
grid_search.best_estimator_.steps
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-45-ec1bd0e775ab> in <module>()
    ----> 1 grid_search.best_estimator_.steps
    

    AttributeError: 'GridSearchCV' object has no attribute 'best_estimator_'



```python
print(f'Calculated cross-validation accuracy: {grid_search.best_score_} while random_search was {random_search.best_score_}')
```


```python
%%time
best_grid_clf = grid_search.best_estimator_
best_grid_clf.fit(X_train, y_train)
```


```python
imdb_acc(best_grid_clf)
```

# Ensemble Models 

## Voting Ensemble

### Simple Majority (aka Hard Voting)


```python
from sklearn.ensemble import VotingClassifier
```


```python
%%time
voting_clf = VotingClassifier(estimators=[('xtc', xtc_clf), ('rfc', rfc_clf)], voting='hard', n_jobs=-1)
voting_clf.fit(X_train, y_train)
```


```python
hard_voting_acc, _ = imdb_acc(voting_clf)
hard_voting_acc
```

#### Soft Voting


```python
%%time
voting_clf = VotingClassifier(estimators=[('lr', lr_clf), ('mnb', mnb_clf)], voting='soft', n_jobs=-1)
voting_clf.fit(X_train, y_train)
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
soft_voting_acc, _ = imdb_acc(voting_clf)
soft_voting_acc
```


```python
gain_acc = soft_voting_acc - lr_acc
if gain_acc > 0:
    print(f'We see that the soft voting gives us an absolute accuracy gain of {gain_acc*100:.2f}% ')
```

### Weighted Classifiers


```python
%%time
weighted_voting_clf = VotingClassifier(estimators=[('lr', lr_clf), ('lr2', lr_clf),('rf', xtc_clf), ('mnb2', mnb_clf),('mnb', mnb_clf)], voting='soft', n_jobs=-1)
weighted_voting_clf.fit(X_train, y_train)
```

Repeat the experiment with 'hard' voting instead of 'soft' voting. This will tell you how does the voting strategy influence the accuracy of our ensembled classifier. 


```python
weighted_voting_acc, _ = imdb_acc(weighted_voting_clf)
weighted_voting_acc
```


```python
gain_acc = weighted_voting_acc - lr_acc
if gain_acc > 0:
    print(f'We see that the weighted voting gives us an absolute accuracy gain of {gain_acc*100:.2f}%')
```


```python
np.corrcoef(mnb_predictions, lr_predictions)[0][1] # this is too high a correlation
```


```python
%%time
corr_voting_clf = VotingClassifier(estimators=[('lr', lr_clf), ('mnb', mnb_clf)], voting='soft', n_jobs=-1)
corr_voting_clf.fit(X_train, y_train)
corr_acc, _ = imdb_acc(corr_voting_clf)
print(corr_acc)
```


```python
np.corrcoef(dtc_predictions,xtc_predictions )[0][1] # this is looks like a low correlation
```


```python
%%time
low_corr_voting_clf = VotingClassifier(estimators=[('dtc', dtc_clf), ('xtc', xtc_clf)], voting='soft', n_jobs=-1)
low_corr_voting_clf.fit(X_train, y_train)
low_corr_acc, _ = imdb_acc(low_corr_voting_clf)
print(low_corr_acc)
```
