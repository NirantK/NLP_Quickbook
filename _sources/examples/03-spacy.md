
# Leveraging Linguistics

## Getting Started


```python
!conda install -y spacy 
# !pip install spacy
```


```python
!python -m spacy download en_core_web_lg
```

If there is an error above, you can use the smaller model as well. 

Try:
- Windows Shell:```python -m spacy download en``` as **Administrator**
- Linux Terminal:```sudo python -m spacy download en ```


```python
import spacy
from spacy import displacy # for visualization
nlp = spacy.load('en_core_web_lg')
```


```python
spacy.__version__
```




    '2.0.11'



**Introducing textacy**:


```python
!conda install -c conda-forge textacy 
# !pip install textacy
```


```python
import textacy
```

## Redacting Names with Named Entity Recognition


```python
text = "Madam Pomfrey, the nurse, was kept busy by a sudden spate of colds among the staff and students. Her Pepperup potion worked instantly, though it left the drinker smoking at the ears for several hours afterward. Ginny Weasley, who had been looking pale, was bullied into taking some by Percy."
```


```python
# Parse the text with spaCy. This runs the entire NLP pipeline.
doc = nlp(text)
```

'doc' now contains a parsed version of text. We can use it to do anything we want!. 
For example, this will print out all the named entities that were detected:


```python
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")
```

    Pomfrey (PERSON)
    Pepperup (ORG)
    several hours (TIME)
    Ginny Weasley (PERSON)
    Percy (PERSON)



```python
doc.ents
```




    (Pomfrey, Pepperup, several hours, Ginny Weasley, Percy)




```python
entity.label, entity.label_
```




    (378, 'PERSON')



In spaCy, all human readable labels etc can also be explained using the simple spacy.explain(label) syntax:


```python
spacy.explain('GPE')
```




    'Countries, cities, states'




```python
def redact_names(text):
    doc = nlp(text)
    redacted_sentence = []
    for token in doc:
        if token.ent_type_ == "PERSON":
            redacted_sentence.append("[REDACTED]")
        else:
            redacted_sentence.append(token.string)
    return "".join(redacted_sentence)
```


```python
redact_names(text)
```




    'Madam [REDACTED], the nurse, was kept busy by a sudden spate of colds among the staff and students. Her Pepperup potion worked instantly, though it left the drinker smoking at the ears for several hours afterward. [REDACTED][REDACTED], who had been looking pale, was bullied into taking some by [REDACTED].'




```python
def redact_names(text):
    doc = nlp(text)
    redacted_sentence = []
    for ent in doc.ents:
        ent.merge()
    for token in doc:
        if token.ent_type_ == "PERSON":
            redacted_sentence.append("[REDACTED]")
        else:
            redacted_sentence.append(token.string)
    return "".join(redacted_sentence)
```


```python
redact_names(text)
```




    'Madam [REDACTED], the nurse, was kept busy by a sudden spate of colds among the staff and students. Her Pepperup potion worked instantly, though it left the drinker smoking at the ears for several hours afterward. [REDACTED], who had been looking pale, was bullied into taking some by [REDACTED].'



### Entity Types 

Let's look at some examples of above in real world sentences, we will also use the `spacy.explain()` on all entities to build a quick mental model of how these things work. 


```python
def explain_text_entities(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f'{ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')
```


```python
explain_text_entities('Tesla has gained 20% market share in the months since')
```

    Tesla, Label: ORG, Companies, agencies, institutions, etc.
    20%, Label: PERCENT, Percentage, including "%"
    the months, Label: DATE, Absolute or relative dates or periods



```python
explain_text_entities('Taj Mahal built by Mughal Emperor Shah Jahan stands tall on the banks of Yamuna in modern day Agra, India')
```

    Taj Mahal, Label: PERSON, People, including fictional
    Mughal, Label: NORP, Nationalities or religious or political groups
    Shah Jahan, Label: PERSON, People, including fictional
    Yamuna, Label: LOC, Non-GPE locations, mountain ranges, bodies of water
    Agra, Label: GPE, Countries, cities, states
    India, Label: GPE, Countries, cities, states



```python
explain_text_entities('Ashoka was a great Indian king')
```

    Ashoka, Label: PERSON, People, including fictional
    Indian, Label: NORP, Nationalities or religious or political groups



```python
explain_text_entities('The Ashoka University sponsors the Young India Fellowship')
```

    Ashoka University, Label: ORG, Companies, agencies, institutions, etc.
    the Young India Fellowship, Label: ORG, Companies, agencies, institutions, etc.


## Automatic Question Generation

### Part-of-Speech Tagging


```python
example_text = 'Bansoori is an Indian classical instrument. Tom plays Bansoori and Guitar.'
```


```python
doc = nlp(example_text)
```

We need noun chunks. Noun chunks are _noun phrases_ - not a single word, but a short phrase which describes the noun. For example, "the blue skies" or "the world’s largest conglomerate". 

To get the noun chunks in a document, simply iterate over `doc.noun_chunks`: 


```python
for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f'sentence{idx+1}', noun)
```

    sentence1 Bansoori
    sentence1 an Indian classical instrument
    sentence2 Tom
    sentence2 Bansoori
    sentence2 Guitar


Our example text has two sentences, we can pull out noun phrase chunks from each sentence. We pull out noun phrases instead of single words. This means, we are able to pull out 'an Indian classical instrument' as one noun. This is quite useful as we will see in a moment.  

Next, let's take a quick look at all parts-of-speech tags in our example text. We will use the verbs and adjectives to write some simple question generating logic. 


```python
for token in doc:
    print(token, token.pos_, token.tag_)
```

    Bansoori PROPN NNP
    is VERB VBZ
    an DET DT
    Indian ADJ JJ
    classical ADJ JJ
    instrument NOUN NN
    . PUNCT .
    Tom PROPN NNP
    plays VERB VBZ
    Bansoori PROPN NNP
    and CCONJ CC
    Guitar PROPN NNP
    . PUNCT .


Notice that here 'instrument' is tagged as a NOUN while 'Indian' and 'classical' are tagged as adjectives. This makes sense. Addititionally, Bansoori and Guitar are tagged as PROPN or Proper Nouns. 

**Nouns vs Proper Noun** 
Nouns name people, places, and things. Common nouns name general items like waiter, jeans, country. Proper nouns name specific things like Roger, Levi's, India

### Creating a Ruleset

Quite often when using linguistics, you will be writing custom rules. Here is one data structure suggestion to help you store these rules: list of dictionaries. Each dictionary in turn can have elements ranging from simple string lists to lists to strings. Avoid nesting a list of dictionaries inside a dictionary:


```python
ruleset = [
    {
        'id': 1, 
        'req_tags': ['NNP', 'VBZ', 'NN'],
    }, 
    {
        'id': 2, 
        'req_tags': ['NNP', 'VBZ'],
    }
    ]
```

Here, I have written two rules. Each rule is simply a collection of part-of-speech tags stored under the 'req_tags' key. Each rule comprises of all the tags that I will look for in a particular sentence. 

Depending on 'id', I will use a hard coded question template to generate my questions. In practice, you can and should move the question template to your ruleset.  


```python
print(ruleset)
```

    [{'id': 1, 'req_tags': ['NNP', 'VBZ', 'NN']}, {'id': 2, 'req_tags': ['NNP', 'VBZ']}]


Next, I need a function to pull out all tokens which match a particular tag. We do this by simply iterating over the entire list of and matching each token against the target tag. 


```python
def get_pos_tag(doc, tag):
    return [tok for tok in doc if tok.tag_ == tag]
```


```python
def sent_to_ques(sent:str)->str:
    """
    Return a question string corresponding to a sentence string using a set of pre-written rules
    """
    doc = nlp(sent)
    pos_tags = [token.tag_ for token in doc]
    for idx, rule in enumerate(ruleset):
        if rule['id'] == 1:
            if all(key in pos_tags for key in rule['req_tags']): 
                print(f"Rule id {rule['id']} matched for sentence: {sent}")
                NNP = get_pos_tag(doc, "NNP")
                NNP = str(NNP[0])
                VBZ = get_pos_tag(doc, "VBZ")
                VBZ = str(VBZ[0])
                ques = f'What {VBZ} {NNP}?'
                return(ques)
        if rule['id'] == 2:
            if all(key in pos_tags for key in rule['req_tags']): #'NNP', 'VBZ' in sentence.
                print(f"Rule id {rule['id']} matched for sentence: {sent}")
                NNP = get_pos_tag(doc, "NNP")
                NNP = str(NNP[0])
                VBZ = get_pos_tag(doc, "VBZ")
                VBZ = str(VBZ[0].lemma_)
                ques = f'What does {NNP} {VBZ}?'
                return(ques)
```


```python
for sent in doc.sents:
    print(f"The generated question is: {sent_to_ques(str(sent))}")
```

    Rule id 1 matched for sentence: Bansoori is an Indian classical instrument.
    The generated question is: What is Bansoori?
    Rule id 2 matched for sentence: Tom plays Bansoori and Guitar.
    The generated question is: What does Tom play?


# Question Generation using Dependency Parsing


```python
for token in doc:
    print(token, token.dep_)
```

    Bansoori nsubj
    is ROOT
    an det
    Indian amod
    classical amod
    instrument attr
    . punct
    Tom nsubj
    plays ROOT
    Bansoori dobj
    and cc
    Guitar conj
    . punct



```python
for token in doc:
    print(token, token.dep_, spacy.explain(token.dep_))
```

    Bansoori nsubj nominal subject
    is ROOT None
    an det determiner
    Indian amod adjectival modifier
    classical amod adjectival modifier
    instrument attr attribute
    . punct punctuation
    Tom nsubj nominal subject
    plays ROOT None
    Bansoori dobj direct object
    and cc coordinating conjunction
    Guitar conj conjunct
    . punct punctuation


## Visualizing the Relationship

spaCy has an inbuilt tool called displacy for displaying simple, but clean and powerful visualizations. It offers two primary modes: Named Entity Recognition and Dependency Parsing. Here we will use the 'dep' or dependency mode. 


```python
displacy.render(doc, style='dep', jupyter=True)
```


<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" id="0" class="displacy" width="1975" height="487.0" style="max-width: none; height: 487.0px; color: #000000; background: #ffffff; font-family: Arial"><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="50">Bansoori</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">PROPN</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="225">is</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">VERB</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="400">an</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">DET</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="575">Indian</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">ADJ</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="750">classical</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="750">ADJ</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="925">instrument.</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="925">NOUN</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="1100">Tom</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="1100">PROPN</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="1275">plays</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="1275">VERB</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="1450">Bansoori</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="1450">PROPN</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="1625">and</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="1625">CCONJ</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0"><tspan class="displacy-word" fill="currentColor" x="1800">Guitar.</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="1800">PROPN</tspan></text><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-0" stroke-width="2px" d="M70,352.0 C70,264.5 210.0,264.5 210.0,352.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-0" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">nsubj</textPath></text><path class="displacy-arrowhead" d="M70,354.0 L62,342.0 78,342.0" fill="currentColor"/></g><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-1" stroke-width="2px" d="M420,352.0 C420,89.5 920.0,89.5 920.0,352.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-1" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">det</textPath></text><path class="displacy-arrowhead" d="M420,354.0 L412,342.0 428,342.0" fill="currentColor"/></g><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-2" stroke-width="2px" d="M595,352.0 C595,177.0 915.0,177.0 915.0,352.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-2" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">amod</textPath></text><path class="displacy-arrowhead" d="M595,354.0 L587,342.0 603,342.0" fill="currentColor"/></g><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-3" stroke-width="2px" d="M770,352.0 C770,264.5 910.0,264.5 910.0,352.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-3" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">amod</textPath></text><path class="displacy-arrowhead" d="M770,354.0 L762,342.0 778,342.0" fill="currentColor"/></g><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-4" stroke-width="2px" d="M245,352.0 C245,2.0 925.0,2.0 925.0,352.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-4" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">attr</textPath></text><path class="displacy-arrowhead" d="M925.0,354.0 L933.0,342.0 917.0,342.0" fill="currentColor"/></g><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-5" stroke-width="2px" d="M1120,352.0 C1120,264.5 1260.0,264.5 1260.0,352.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-5" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">nsubj</textPath></text><path class="displacy-arrowhead" d="M1120,354.0 L1112,342.0 1128,342.0" fill="currentColor"/></g><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-6" stroke-width="2px" d="M1295,352.0 C1295,264.5 1435.0,264.5 1435.0,352.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-6" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">dobj</textPath></text><path class="displacy-arrowhead" d="M1435.0,354.0 L1443.0,342.0 1427.0,342.0" fill="currentColor"/></g><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-7" stroke-width="2px" d="M1470,352.0 C1470,264.5 1610.0,264.5 1610.0,352.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-7" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">cc</textPath></text><path class="displacy-arrowhead" d="M1610.0,354.0 L1618.0,342.0 1602.0,342.0" fill="currentColor"/></g><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-8" stroke-width="2px" d="M1470,352.0 C1470,177.0 1790.0,177.0 1790.0,352.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-8" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">conj</textPath></text><path class="displacy-arrowhead" d="M1790.0,354.0 L1798.0,342.0 1782.0,342.0" fill="currentColor"/></g></svg>



```python
tricky_doc = nlp('This is ship-shipping ship, shipping shipping ships')
```


```python
displacy.render(tricky_doc, style='dep', jupyter=True)
```


<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" id="0" class="displacy" width="1450" height="487.0" style="max-width: none; height: 487.0px; color: #000000; background: #ffffff; font-family: Arial">
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="50">This</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">DET</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="225">is</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="400">ship-</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">NOUN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="575">shipping</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">NOUN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="750">ship,</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="750">NOUN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="925">shipping</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="925">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="1100">shipping</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1100">NOUN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="397.0">
    <tspan class="displacy-word" fill="currentColor" x="1275">ships</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1275">NOUN</tspan>
</text>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-0-0" stroke-width="2px" d="M70,352.0 C70,264.5 210.0,264.5 210.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-0-0" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M70,354.0 L62,342.0 78,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-0-1" stroke-width="2px" d="M420,352.0 C420,264.5 560.0,264.5 560.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-0-1" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">npadvmod</textPath>
    </text>
    <path class="displacy-arrowhead" d="M420,354.0 L412,342.0 428,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-0-2" stroke-width="2px" d="M595,352.0 C595,264.5 735.0,264.5 735.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-0-2" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">compound</textPath>
    </text>
    <path class="displacy-arrowhead" d="M595,354.0 L587,342.0 603,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-0-3" stroke-width="2px" d="M245,352.0 C245,89.5 745.0,89.5 745.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-0-3" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">attr</textPath>
    </text>
    <path class="displacy-arrowhead" d="M745.0,354.0 L753.0,342.0 737.0,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-0-4" stroke-width="2px" d="M245,352.0 C245,2.0 925.0,2.0 925.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-0-4" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">advcl</textPath>
    </text>
    <path class="displacy-arrowhead" d="M925.0,354.0 L933.0,342.0 917.0,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-0-5" stroke-width="2px" d="M1120,352.0 C1120,264.5 1260.0,264.5 1260.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-0-5" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">compound</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1120,354.0 L1112,342.0 1128,342.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-0-6" stroke-width="2px" d="M945,352.0 C945,177.0 1265.0,177.0 1265.0,352.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-0-6" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">dobj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1265.0,354.0 L1273.0,342.0 1257.0,342.0" fill="currentColor"/>
</g>
</svg>



```python
from textacy.spacier import utils as spacy_utils
```


```python
??spacy_utils.get_main_verbs_of_sent
```


```python
# Signature: spacy_utils.get_main_verbs_of_sent(sent)
# Source:   
# def get_main_verbs_of_sent(sent):
#     """Return the main (non-auxiliary) verbs in a sentence."""
#     return [tok for tok in sent
#             if tok.pos == VERB and tok.dep_ not in constants.AUX_DEPS]
# File:      d:\miniconda3\envs\nlp\lib\site-packages\textacy\spacier\utils.py
# Type:      function
```


```python
toy_sentence = 'Shivangi is an engineer'
doc = nlp(toy_sentence)
```

What are the entities in this sentence? 


```python
displacy.render(doc, style='ent', jupyter=True)
```


<div class="entities" style="line-height: 2.5">
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    Shivangi
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 is an engineer</div>


Let's find out the main verb in this sentence: 


```python
verbs = spacy_utils.get_main_verbs_of_sent(doc)
print(verbs)
```

    [is]


And what are nominal subjects of this verb?   


```python
for verb in verbs:
    print(verb, spacy_utils.get_subjects_of_verb(verb))
```

    is [Shivangi]


You will notice that this has a reasonable overlap with the noun phrases which we pulled from our part-of-speech tagging but can be different as well. 


```python
[(token, token.tag_) for token in doc]
```




    [(Shivangi, 'NNP'), (is, 'VBZ'), (an, 'DT'), (engineer, 'NN')]



Tip: As an exercise, extend this approach to at least add Who, Where and When questions as practice. 

# Level Up: Question and Answer
So far, we have been trying to generate questions. But if you were trying to make an automated quiz for students, you would also need to mine the right answer. 

The answer in this case will be simply the objects of verb. What is an object of verb? 

> In the sentence, "Give the book to me," "book" is the direct object of the verb "give," and "me" is the indirect object. - from the Cambridge English Dictionary

Loosely, object is the piece on which our verb acts. This is almost always the answer to our "what". Let's write a question to find the objects of any verb --- or wait, we can pull it from the `textacy.spacier.utils`. 


```python
spacy_utils.get_objects_of_verb(verb)
```




    [engineer]




```python
for verb in verbs:
    print(verb, spacy_utils.get_objects_of_verb(verb))
```

    is [engineer]



```python
displacy.render(doc, style='dep', jupyter=True)
```


<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" id="0" class="displacy" width="750" height="312.0" style="max-width: none; height: 312.0px; color: #000000; background: #ffffff; font-family: Arial"><text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0"><tspan class="displacy-word" fill="currentColor" x="50">Shivangi</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">PROPN</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0"><tspan class="displacy-word" fill="currentColor" x="225">is</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">VERB</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0"><tspan class="displacy-word" fill="currentColor" x="400">an</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">DET</tspan></text><text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0"><tspan class="displacy-word" fill="currentColor" x="575">engineer</tspan><tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">NOUN</tspan></text><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-0" stroke-width="2px" d="M70,177.0 C70,89.5 220.0,89.5 220.0,177.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-0" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">nsubj</textPath></text><path class="displacy-arrowhead" d="M70,179.0 L62,167.0 78,167.0" fill="currentColor"/></g><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-1" stroke-width="2px" d="M420,177.0 C420,89.5 570.0,89.5 570.0,177.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-1" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">det</textPath></text><path class="displacy-arrowhead" d="M420,179.0 L412,167.0 428,167.0" fill="currentColor"/></g><g class="displacy-arrow"><path class="displacy-arc" id="arrow-0-2" stroke-width="2px" d="M245,177.0 C245,2.0 575.0,2.0 575.0,177.0" fill="none" stroke="currentColor"/><text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px"><textPath xlink:href="#arrow-0-2" class="displacy-label" startOffset="50%" fill="currentColor" text-anchor="middle">attr</textPath></text><path class="displacy-arrowhead" d="M575.0,179.0 L583.0,167.0 567.0,167.0" fill="currentColor"/></g></svg>


Let's look at the output of our functions for the example text. The first is the sentence itself, then the root verb, than the lemma form of that verb, followed by subjects of the verb and then objects.  


```python
doc = nlp(example_text)
for sentence in doc.sents:
    print(sentence, sentence.root, sentence.root.lemma_, spacy_utils.get_subjects_of_verb(sentence.root), spacy_utils.get_objects_of_verb(sentence.root))
```

    Bansoori is an Indian classical instrument. is be [Bansoori] [instrument]
    Tom plays Bansoori and Guitar. plays play [Tom] [Bansoori, Guitar]


Let's arrange the pieces above into a neat function which we can then re-use


```python
def para_to_ques(eg_text):
    doc = nlp(eg_text)
    results = []
    for sentence in doc.sents:
        root = sentence.root
        ask_about = spacy_utils.get_subjects_of_verb(root)
        answers = spacy_utils.get_objects_of_verb(root)
        if len(ask_about) > 0 and len(answers) > 0:
            if root.lemma_ == "be":
                question = f'What {root} {ask_about[0]}?'
            else:
                question = f'What does {ask_about[0]} {root.lemma_}?'
            results.append({'question':question, 'answers':answers})
    return results
```


```python
para_to_ques(example_text)
```




    [{'question': 'What is Bansoori?', 'answers': [instrument]},
     {'question': 'What does Tom play?', 'answers': [Bansoori, Guitar]}]



This seems right to me. Let's run this on a larger sample of sentences. This sample has varying degrees of complexities and sentence structures. 


```python
large_example_text = """
Puliyogare is a South Indian dish made of rice and tamarind. 
Priya writes poems. Shivangi bakes cakes. Sachin sings in the orchestra.

Osmosis is the movement of a solvent across a semipermeable membrane toward a higher concentration of solute. In biological systems, the solvent is typically water, but osmosis can occur in other liquids, supercritical liquids, and even gases.
When a cell is submerged in water, the water molecules pass through the cell membrane from an area of low solute concentration to high solute concentration. For example, if the cell is submerged in saltwater, water molecules move out of the cell. If a cell is submerged in freshwater, water molecules move into the cell.

Raja-Yoga is divided into eight steps. The first is Yama. Yama is nonviolence, truthfulness, continence, and non-receiving of any gifts.
After Yama, Raja-Yoga has Niyama. cleanliness, contentment, austerity, study, and self - surrender to God.
The steps are Yama and Niyama. 
"""

```


```python
para_to_ques(large_example_text)
```




    [{'question': 'What is Puliyogare?', 'answers': [dish]},
     {'question': 'What does Priya write?', 'answers': [poems]},
     {'question': 'What does Shivangi bake?', 'answers': [cakes]},
     {'question': 'What is Osmosis?', 'answers': [movement]},
     {'question': 'What is solvent?', 'answers': [water]},
     {'question': 'What is first?', 'answers': [Yama]},
     {'question': 'What is Yama?',
      'answers': [nonviolence, truthfulness, continence, of]},
     {'question': 'What does Yoga have?', 'answers': [Niyama]},
     {'question': 'What are steps?', 'answers': [Yama, Niyama]}]



# Facts Extraction using Semi Structured Sentence Parsing
Introducing textacy,

Boss mode with co reference resolution
