

```python
import gensim
print(f'gensim: {gensim.__version__}')
```

    D:\Miniconda3\envs\nlp\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")


    gensim: 3.4.0


Let's download some pre-trained GLove embeddings: 


```python
!conda install -y tqdm
```

    Solving environment: ...working... done
    
    # All requested packages already installed.
    



```python
from tqdm import tqdm
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)

def get_data(url, filename):
    """
    Download data if the filename does not exist already
    Uses Tqdm to show download progress
    """
    import os
    from urllib.request import urlretrieve
    
    if not os.path.exists(filename):

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename, reporthook=t.update_to)
```


```python
embedding_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
get_data(embedding_url, 'data/glove.6B.zip')
```


```python
# # We need to run this only once, can unzip manually unzip to the data directory too
!unzip data/glove.6B.zip 
!mv glove.6B.300d.txt data/glove.6B.300d.txt 
!mv glove.6B.200d.txt data/glove.6B.200d.txt 
!mv glove.6B.100d.txt data/glove.6B.100d.txt 
!mv glove.6B.50d.txt data/glove.6B.50d.txt 
```

    Archive:  data/glove.6B.zip
      inflating: glove.6B.50d.txt        
      inflating: glove.6B.100d.txt       
      inflating: glove.6B.200d.txt       
      inflating: glove.6B.300d.txt       



```python
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'data/glove.6B.300d.txt'

word2vec_output_file = 'data/glove.6B.300d.word2vec.txt'
```


```python
import os
if not os.path.exists(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)
```

### KeyedVectors API


```python
from gensim.models import KeyedVectors
filename = word2vec_output_file 
```


```python
%%time
# load the Stanford GloVe model from file, this is Disk I/O and can be slow
pretrained_w2v_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# binary=False format for human readable text (.txt) files, and binary=True for .bin files 
```

    Wall time: 1min 24s



```python
# calculate: (king - man) + woman = ?
result = pretrained_w2v_model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)
```

    D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel_launcher.py:2: DeprecationWarning: Call to deprecated `wv` (Attribute will be removed in 4.0.0, use self instead).
      


    [('queen', 0.6713277101516724)]



```python
# calculate: (india - canada) +  = ?
result = pretrained_w2v_model.most_similar(positive=['quora', 'facebook'], negative=['linkedin'], topn=1)
print(result)
```

    [('twitter', 0.37966805696487427)]



```python
pretrained_w2v_model.most_similar('india')
```




    [('indian', 0.7355823516845703),
     ('pakistan', 0.7285579442977905),
     ('delhi', 0.6846905946731567),
     ('bangladesh', 0.620319128036499),
     ('lanka', 0.609517514705658),
     ('sri', 0.6011613607406616),
     ('kashmir', 0.5746493935585022),
     ('nepal', 0.5421023964881897),
     ('pradesh', 0.5405810475349426),
     ('maharashtra', 0.518537700176239)]



#### What is missing in both word2vec and GloVe? 


```python
try:
    pretrained_w2v_model.wv.most_similar('nirant')
except Exception as e:
    print(e)
```

    D:\Miniconda3\envs\nlp\lib\site-packages\ipykernel_launcher.py:2: DeprecationWarning: Call to deprecated `wv` (Attribute will be removed in 4.0.0, use self instead).
      


    "word 'nirant' not in vocabulary"


### How to handle OOV words? 


```python
ted_dataset = "https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip"
get_data(ted_dataset, "data/ted_en.zip")
```


```python
import zipfile
import lxml.etree
with zipfile.ZipFile('data/ted_en.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))
```


```python
input_text[:500]
```




    "Here are two reasons companies fail: they only do more of the same, or they only do what's new.\nTo me the real, real solution to quality growth is figuring out the balance between two activities: exploration and exploitation. Both are necessary, but it can be too much of a good thing.\nConsider Facit. I'm actually old enough to remember them. Facit was a fantastic company. They were born deep in the Swedish forest, and they made the best mechanical calculators in the world. Everybody used them. A"




```python
import re
# remove parenthesis 
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)

# store as list of sentences
sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)

# store as list of lists of words
sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)
```


```python
print(sentences_ted[:2])
```

    [['here', 'are', 'two', 'reasons', 'companies', 'fail', 'they', 'only', 'do', 'more', 'of', 'the', 'same', 'or', 'they', 'only', 'do', 'what', 's', 'new'], ['to', 'me', 'the', 'real', 'real', 'solution', 'to', 'quality', 'growth', 'is', 'figuring', 'out', 'the', 'balance', 'between', 'two', 'activities', 'exploration', 'and', 'exploitation']]



```python
import json
with open('ted_clean_sentences.json', 'w') as fp:
    json.dump(sentences_ted, fp)
```


```python
with open('ted_clean_sentences.json', 'r') as fp:
    sentences_ted = json.load(fp)
```


```python
print(sentences_ted[:2])
```

    [['here', 'are', 'two', 'reasons', 'companies', 'fail', 'they', 'only', 'do', 'more', 'of', 'the', 'same', 'or', 'they', 'only', 'do', 'what', 's', 'new'], ['to', 'me', 'the', 'real', 'real', 'solution', 'to', 'quality', 'growth', 'is', 'figuring', 'out', 'the', 'balance', 'between', 'two', 'activities', 'exploration', 'and', 'exploitation']]


### Train FastText Embedddings


```python
from gensim.models.fasttext import FastText
```


```python
%%time
fasttext_ted_model = FastText(sentences_ted, size=100, window=5, min_count=5, workers=-1, sg=1)
# sg = 1 denotes skipgram, else CBOW is used
```

    Wall time: 5.48 s



```python
fasttext_ted_model.wv.most_similar("india")
```




    [('indians', 0.5911639928817749),
     ('indian', 0.5406097769737244),
     ('indiana', 0.4898717999458313),
     ('indicated', 0.44004374742507935),
     ('indicate', 0.4042605757713318),
     ('internal', 0.39166826009750366),
     ('interior', 0.3871103823184967),
     ('byproducts', 0.37529298663139343),
     ('princesses', 0.37265270948410034),
     ('indications', 0.369659960269928)]



### Train word2vec Embeddings


```python
from gensim.models.word2vec import Word2Vec
```


```python
%%time
word2vec_ted_model = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=-1, sg=1)
```

    Wall time: 1.44 s



```python
word2vec_ted_model.wv.most_similar("india")
```




    [('bordered', 0.41709238290786743),
     ('hovering', 0.4083016514778137),
     ('almost', 0.3865964710712433),
     ('sad', 0.3704090118408203),
     ('supporters', 0.3616541624069214),
     ('spite', 0.3598758280277252),
     ('wrinkles', 0.3590206205844879),
     ('guaranteed', 0.3535975515842438),
     ('hd', 0.3512127995491028),
     ('assistant', 0.346971333026886)]



## fastText or word2vec? 

# Document Embeddings


```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
from pprint import pprint
import multiprocessing
```


```python
import zipfile
import lxml.etree
with zipfile.ZipFile('data/ted_en.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
    
talks = doc.xpath('//content/text()')
```


```python
def read_corpus(talks, tokens_only=False):
    for i, line in enumerate(talks):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])
```


```python
read_corpus(talks)
```




    <generator object read_corpus at 0x00000218240FE990>




```python
ted_talk_docs = list(read_corpus(talks)) 
```


```python
ted_talk_docs[0]
```




    TaggedDocument(words=['here', 'are', 'two', 'reasons', 'companies', 'fail', 'they', 'only', 'do', 'more', 'of', 'the', 'same', 'or', 'they', 'only', 'do', 'what', 'new', 'to', 'me', 'the', 'real', 'real', 'solution', 'to', 'quality', 'growth', 'is', 'figuring', 'out', 'the', 'balance', 'between', 'two', 'activities', 'exploration', 'and', 'exploitation', 'both', 'are', 'necessary', 'but', 'it', 'can', 'be', 'too', 'much', 'of', 'good', 'thing', 'consider', 'facit', 'actually', 'old', 'enough', 'to', 'remember', 'them', 'facit', 'was', 'fantastic', 'company', 'they', 'were', 'born', 'deep', 'in', 'the', 'swedish', 'forest', 'and', 'they', 'made', 'the', 'best', 'mechanical', 'calculators', 'in', 'the', 'world', 'everybody', 'used', 'them', 'and', 'what', 'did', 'facit', 'do', 'when', 'the', 'electronic', 'calculator', 'came', 'along', 'they', 'continued', 'doing', 'exactly', 'the', 'same', 'in', 'six', 'months', 'they', 'went', 'from', 'maximum', 'revenue', 'and', 'they', 'were', 'gone', 'gone', 'to', 'me', 'the', 'irony', 'about', 'the', 'facit', 'story', 'is', 'hearing', 'about', 'the', 'facit', 'engineers', 'who', 'had', 'bought', 'cheap', 'small', 'electronic', 'calculators', 'in', 'japan', 'that', 'they', 'used', 'to', 'double', 'check', 'their', 'calculators', 'laughter', 'facit', 'did', 'too', 'much', 'exploitation', 'but', 'exploration', 'can', 'go', 'wild', 'too', 'few', 'years', 'back', 'worked', 'closely', 'alongside', 'european', 'biotech', 'company', 'let', 'call', 'them', 'oncosearch', 'the', 'company', 'was', 'brilliant', 'they', 'had', 'applications', 'that', 'promised', 'to', 'diagnose', 'even', 'cure', 'certain', 'forms', 'of', 'blood', 'cancer', 'every', 'day', 'was', 'about', 'creating', 'something', 'new', 'they', 'were', 'extremely', 'innovative', 'and', 'the', 'mantra', 'was', 'when', 'we', 'only', 'get', 'it', 'right', 'or', 'even', 'we', 'want', 'it', 'perfect', 'the', 'sad', 'thing', 'is', 'before', 'they', 'became', 'perfect', 'even', 'good', 'enough', 'they', 'became', 'obsolete', 'oncosearch', 'did', 'too', 'much', 'exploration', 'first', 'heard', 'about', 'exploration', 'and', 'exploitation', 'about', 'years', 'ago', 'when', 'worked', 'as', 'visiting', 'scholar', 'at', 'stanford', 'university', 'the', 'founder', 'of', 'the', 'idea', 'is', 'jim', 'march', 'and', 'to', 'me', 'the', 'power', 'of', 'the', 'idea', 'is', 'its', 'practicality', 'exploration', 'exploration', 'is', 'about', 'coming', 'up', 'with', 'what', 'new', 'it', 'about', 'search', 'it', 'about', 'discovery', 'it', 'about', 'new', 'products', 'it', 'about', 'new', 'innovations', 'it', 'about', 'changing', 'our', 'frontiers', 'our', 'heroes', 'are', 'people', 'who', 'have', 'done', 'exploration', 'madame', 'curie', 'picasso', 'neil', 'armstrong', 'sir', 'edmund', 'hillary', 'etc', 'come', 'from', 'norway', 'all', 'our', 'heroes', 'are', 'explorers', 'and', 'they', 'deserve', 'to', 'be', 'we', 'all', 'know', 'that', 'exploration', 'is', 'risky', 'we', 'don', 'know', 'the', 'answers', 'we', 'don', 'know', 'if', 'we', 're', 'going', 'to', 'find', 'them', 'and', 'we', 'know', 'that', 'the', 'risks', 'are', 'high', 'exploitation', 'is', 'the', 'opposite', 'exploitation', 'is', 'taking', 'the', 'knowledge', 'we', 'have', 'and', 'making', 'good', 'better', 'exploitation', 'is', 'about', 'making', 'our', 'trains', 'run', 'on', 'time', 'it', 'about', 'making', 'good', 'products', 'faster', 'and', 'cheaper', 'exploitation', 'is', 'not', 'risky', 'in', 'the', 'short', 'term', 'but', 'if', 'we', 'only', 'exploit', 'it', 'very', 'risky', 'in', 'the', 'long', 'term', 'and', 'think', 'we', 'all', 'have', 'memories', 'of', 'the', 'famous', 'pop', 'groups', 'who', 'keep', 'singing', 'the', 'same', 'songs', 'again', 'and', 'again', 'until', 'they', 'become', 'obsolete', 'or', 'even', 'pathetic', 'that', 'the', 'risk', 'of', 'exploitation', 'so', 'if', 'we', 'take', 'long', 'term', 'perspective', 'we', 'explore', 'if', 'we', 'take', 'short', 'term', 'perspective', 'we', 'exploit', 'small', 'children', 'they', 'explore', 'all', 'day', 'all', 'day', 'it', 'about', 'exploration', 'as', 'we', 'grow', 'older', 'we', 'explore', 'less', 'because', 'we', 'have', 'more', 'knowledge', 'to', 'exploit', 'on', 'the', 'same', 'goes', 'for', 'companies', 'companies', 'become', 'by', 'nature', 'less', 'innovative', 'as', 'they', 'become', 'more', 'competent', 'and', 'this', 'is', 'of', 'course', 'big', 'worry', 'to', 'ceos', 'and', 'hear', 'very', 'often', 'questions', 'phrased', 'in', 'different', 'ways', 'for', 'example', 'how', 'can', 'both', 'effectively', 'run', 'and', 'reinvent', 'my', 'company', 'or', 'how', 'can', 'make', 'sure', 'that', 'our', 'company', 'changes', 'before', 'we', 'become', 'obsolete', 'or', 'are', 'hit', 'by', 'crisis', 'so', 'doing', 'one', 'well', 'is', 'difficult', 'doing', 'both', 'well', 'as', 'the', 'same', 'time', 'is', 'art', 'pushing', 'both', 'exploration', 'and', 'exploitation', 'so', 'one', 'thing', 'we', 've', 'found', 'is', 'only', 'about', 'two', 'percent', 'of', 'companies', 'are', 'able', 'to', 'effectively', 'explore', 'and', 'exploit', 'at', 'the', 'same', 'time', 'in', 'parallel', 'but', 'when', 'they', 'do', 'the', 'payoffs', 'are', 'huge', 'so', 'we', 'have', 'lots', 'of', 'great', 'examples', 'we', 'have', 'nestlé', 'creating', 'nespresso', 'we', 'have', 'lego', 'going', 'into', 'animated', 'films', 'toyota', 'creating', 'the', 'hybrids', 'unilever', 'pushing', 'into', 'sustainability', 'there', 'are', 'lots', 'of', 'examples', 'and', 'the', 'benefits', 'are', 'huge', 'why', 'is', 'balancing', 'so', 'difficult', 'think', 'it', 'difficult', 'because', 'there', 'are', 'so', 'many', 'traps', 'that', 'keep', 'us', 'where', 'we', 'are', 'so', 'll', 'talk', 'about', 'two', 'but', 'there', 'are', 'many', 'so', 'let', 'talk', 'about', 'the', 'perpetual', 'search', 'trap', 'we', 'discover', 'something', 'but', 'we', 'don', 'have', 'the', 'patience', 'or', 'the', 'persistence', 'to', 'get', 'at', 'it', 'and', 'make', 'it', 'work', 'so', 'instead', 'of', 'staying', 'with', 'it', 'we', 'create', 'something', 'new', 'but', 'the', 'same', 'goes', 'for', 'that', 'then', 'we', 're', 'in', 'the', 'vicious', 'circle', 'of', 'actually', 'coming', 'up', 'with', 'ideas', 'but', 'being', 'frustrated', 'oncosearch', 'was', 'good', 'example', 'famous', 'example', 'is', 'of', 'course', 'xerox', 'but', 'we', 'don', 'only', 'see', 'this', 'in', 'companies', 'we', 'see', 'this', 'in', 'the', 'public', 'sector', 'as', 'well', 'we', 'all', 'know', 'that', 'any', 'kind', 'of', 'effective', 'reform', 'of', 'education', 'research', 'health', 'care', 'even', 'defense', 'takes', 'maybe', 'years', 'to', 'work', 'but', 'still', 'we', 'change', 'much', 'more', 'often', 'we', 'really', 'don', 'give', 'them', 'the', 'chance', 'another', 'trap', 'is', 'the', 'success', 'trap', 'facit', 'fell', 'into', 'the', 'success', 'trap', 'they', 'literally', 'held', 'the', 'future', 'in', 'their', 'hands', 'but', 'they', 'couldn', 'see', 'it', 'they', 'were', 'simply', 'so', 'good', 'at', 'making', 'what', 'they', 'loved', 'doing', 'that', 'they', 'wouldn', 'change', 'we', 'are', 'like', 'that', 'too', 'when', 'we', 'know', 'something', 'well', 'it', 'difficult', 'to', 'change', 'bill', 'gates', 'has', 'said', 'success', 'is', 'lousy', 'teacher', 'it', 'seduces', 'us', 'into', 'thinking', 'we', 'cannot', 'fail', 'that', 'the', 'challenge', 'with', 'success', 'so', 'think', 'there', 'are', 'some', 'lessons', 'and', 'think', 'they', 'apply', 'to', 'us', 'and', 'they', 'apply', 'to', 'our', 'companies', 'the', 'first', 'lesson', 'is', 'get', 'ahead', 'of', 'the', 'crisis', 'and', 'any', 'company', 'that', 'able', 'to', 'innovate', 'is', 'actually', 'able', 'to', 'also', 'buy', 'an', 'insurance', 'in', 'the', 'future', 'netflix', 'they', 'could', 'so', 'easily', 'have', 'been', 'content', 'with', 'earlier', 'generations', 'of', 'distribution', 'but', 'they', 'always', 'and', 'think', 'they', 'will', 'always', 'keep', 'pushing', 'for', 'the', 'next', 'battle', 'see', 'other', 'companies', 'that', 'say', 'll', 'win', 'the', 'next', 'innovation', 'cycle', 'whatever', 'it', 'takes', 'second', 'one', 'think', 'in', 'multiple', 'time', 'scales', 'll', 'share', 'chart', 'with', 'you', 'and', 'think', 'it', 'wonderful', 'one', 'any', 'company', 'we', 'look', 'at', 'taking', 'one', 'year', 'perspective', 'and', 'looking', 'at', 'the', 'valuation', 'of', 'the', 'company', 'innovation', 'typically', 'accounts', 'for', 'only', 'about', 'percent', 'so', 'when', 'we', 'think', 'one', 'year', 'innovation', 'isn', 'really', 'that', 'important', 'move', 'ahead', 'take', 'year', 'perspective', 'on', 'the', 'same', 'company', 'suddenly', 'innovation', 'and', 'ability', 'to', 'renew', 'account', 'for', 'percent', 'but', 'companies', 'can', 'choose', 'they', 'need', 'to', 'fund', 'the', 'journey', 'and', 'lead', 'the', 'long', 'term', 'third', 'invite', 'talent', 'don', 'think', 'it', 'possible', 'for', 'any', 'of', 'us', 'to', 'be', 'able', 'to', 'balance', 'exploration', 'and', 'exploitation', 'by', 'ourselves', 'think', 'it', 'team', 'sport', 'think', 'we', 'need', 'to', 'allow', 'challenging', 'think', 'the', 'mark', 'of', 'great', 'company', 'is', 'being', 'open', 'to', 'be', 'challenged', 'and', 'the', 'mark', 'of', 'good', 'corporate', 'board', 'is', 'to', 'constructively', 'challenge', 'think', 'that', 'also', 'what', 'good', 'parenting', 'is', 'about', 'last', 'one', 'be', 'skeptical', 'of', 'success', 'maybe', 'it', 'useful', 'to', 'think', 'back', 'at', 'the', 'old', 'triumph', 'marches', 'in', 'rome', 'when', 'the', 'generals', 'after', 'big', 'victory', 'were', 'given', 'their', 'celebration', 'riding', 'into', 'rome', 'on', 'the', 'carriage', 'they', 'always', 'had', 'companion', 'whispering', 'in', 'their', 'ear', 'remember', 'you', 're', 'only', 'human', 'so', 'hope', 'made', 'the', 'point', 'balancing', 'exploration', 'and', 'exploitation', 'has', 'huge', 'payoff', 'but', 'it', 'difficult', 'and', 'we', 'need', 'to', 'be', 'conscious', 'want', 'to', 'just', 'point', 'out', 'two', 'questions', 'that', 'think', 'are', 'useful', 'first', 'question', 'is', 'looking', 'at', 'your', 'own', 'company', 'in', 'which', 'areas', 'do', 'you', 'see', 'that', 'the', 'company', 'is', 'at', 'the', 'risk', 'of', 'falling', 'into', 'success', 'traps', 'of', 'just', 'going', 'on', 'autopilot', 'and', 'what', 'can', 'you', 'do', 'to', 'challenge', 'second', 'question', 'is', 'when', 'did', 'explore', 'something', 'new', 'last', 'and', 'what', 'kind', 'of', 'effect', 'did', 'it', 'have', 'on', 'me', 'is', 'that', 'something', 'should', 'do', 'more', 'of', 'in', 'my', 'case', 'yes', 'so', 'let', 'me', 'leave', 'you', 'with', 'this', 'whether', 'you', 're', 'an', 'explorer', 'by', 'nature', 'or', 'whether', 'you', 'tend', 'to', 'exploit', 'what', 'you', 'already', 'know', 'don', 'forget', 'the', 'beauty', 'is', 'in', 'the', 'balance', 'thank', 'you', 'applause'], tags=[0])




```python
cores = multiprocessing.cpu_count()
print(cores)
```

    8



```python
model = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, epochs=5, workers=cores)
```


```python
%time model.build_vocab(ted_talk_docs)
```

    Wall time: 1.4 s



```python
sentence_1 = 'Modern medicine has changed the way we think about healthcare, life spans and by extension career and marriage'
```


```python
sentence_2 = 'Modern medicine is not just a boon to the rich, making the raw chemicals behind these is also pollutes the poorest neighborhoods'
```


```python
sentence_3 = 'Modern medicine has changed the way we think about healthcare, and increased life spans, delaying weddings'
```


```python
model.docvecs.similarity_unseen_docs(model, sentence_1.split(), sentence_3.split())
```




    -0.14454556996040863




```python
model.docvecs.similarity_unseen_docs(model, sentence_1.split(), sentence_2.split())
```




    -0.04978240807521571




```python
%time model.train(ted_talk_docs, total_examples=model.corpus_count, epochs=model.epochs)
```

    Wall time: 6.77 s



```python
model.infer_vector(sentence_1.split())
```




    array([ 0.20152442,  0.07655947,  0.04110149, -0.09114903, -0.02466601,
            0.10063498, -0.04590227, -0.16054891, -0.23367156, -0.07714292,
           -0.32246125,  0.10532021,  0.11020374, -0.02373328, -0.06048575,
            0.06041928, -0.20840394,  0.11885054, -0.09653657,  0.02215091,
            0.01846626,  0.06881414, -0.01988592,  0.01138998,  0.06924792,
            0.11989842,  0.09510404,  0.01230403,  0.05453861,  0.05833528,
            0.22496092,  0.06185873,  0.15445319, -0.13073249,  0.1320086 ,
            0.15955518,  0.09083826, -0.262743  ,  0.07112081, -0.12404393,
           -0.07876749, -0.17020509, -0.08309909,  0.20299006, -0.07867863,
           -0.19080839, -0.00371094, -0.2119167 , -0.11631834, -0.12984131,
           -0.11451794,  0.12690201, -0.02519317,  0.23437414, -0.11313629,
            0.06674401, -0.0190409 ,  0.3384525 , -0.13124712, -0.12843844,
           -0.2605964 ,  0.22317892, -0.20078087, -0.05607577, -0.08431446,
           -0.20859231,  0.15535517,  0.0073873 , -0.11435535,  0.16722508,
           -0.0567028 ,  0.23436148, -0.1829926 ,  0.05211424, -0.14246033,
            0.20756294,  0.03515876,  0.10574302, -0.05463392,  0.09465599,
           -0.24758984, -0.04593265, -0.13151605,  0.3317288 , -0.13002025,
           -0.37372893, -0.26798424, -0.27239782, -0.16636257,  0.19000524,
            0.12744325, -0.14971398, -0.11483772, -0.01594907,  0.02319706,
            0.03037767, -0.01439404,  0.08120204, -0.188371  , -0.21033412],
          dtype=float32)




```python
model.docvecs.similarity_unseen_docs(model, sentence_1.split(), sentence_3.split())
```




    0.9073806748252071




```python
model.docvecs.similarity_unseen_docs(model, sentence_1.split(), sentence_2.split())
```




    0.7626341790517841




```python
model.docvecs.similarity_unseen_docs(model, sentence_2.split(), sentence_3.split())
```




    0.8026655396100536



# Model Assessment


```python
ranks = []
for idx in range(len(ted_talk_docs)):
    inferred_vector = model.infer_vector(ted_talk_docs[idx].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(idx)
    ranks.append(rank)
```


```python
import collections
collections.Counter(ranks)  # Results vary due to random seeding + very small corpus
```




    Counter({0: 2080, 3: 1, 4: 1, 1: 1, 2: 1, 6: 1})




```python
doc_slice = ' '.join(ted_talk_docs[idx].words)[:500]
print(f'Document ({idx}): «{doc_slice}»\n')
print(f'SIMILAR/DISSIMILAR DOCS PER MODEL {model}')
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
      doc_slice = ' '.join(ted_talk_docs[sims[index][0]].words)[:500]
      print(f'{label} {sims[index]}: «{doc_slice}»\n')
```

    Document (2084): «if you re here today and very happy that you are you ve all heard about how sustainable development will save us from ourselves however when we re not at ted we are often told that real sustainability policy agenda is just not feasible especially in large urban areas like new york city and that because most people with decision making powers in both the public and the private sector really don feel as though they re in danger the reason why here today in part is because of dog an abandoned puppy»
    
    SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dbow,d100,n5,mc2,s0.001,t8)
    MOST (2084, 0.8938855528831482): «if you re here today and very happy that you are you ve all heard about how sustainable development will save us from ourselves however when we re not at ted we are often told that real sustainability policy agenda is just not feasible especially in large urban areas like new york city and that because most people with decision making powers in both the public and the private sector really don feel as though they re in danger the reason why here today in part is because of dog an abandoned puppy»
    
    MEDIAN (949, 0.40211787819862366): «we conventionally divide space into private and public realms and we know these legal distinctions very well because we ve become experts at protecting our private property and private space but we re less attuned to the nuances of the public what translates generic public space into qualitative space mean this is something that our studio has been working on for the past decade and we re doing this through some case studies large chunk of our work has been put into transforming this neglected i»
    
    LEAST (876, 0.11194153130054474): «so the machine going to talk you about is what call the greatest machine that never was it was machine that was never built and yet it will be built it was machine that was designed long before anyone thought about computers if you know anything about the history of computers you will know that in the and the simple computers were created that started the computer revolution we have today and you would be correct except for you have the wrong century the first computer was really designed in the»
    

