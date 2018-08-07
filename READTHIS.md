# Leveraging Linguistics for Chat

There is immense utility in being able to mine long human texts for particular information or manipulate it in meaningful ways. 

We will take a deep dive of the linguistic features of spaCy (and mention the old Stanford NLTK, CoreNLP for reference). 

We will cover dependency parsing, POS Tagging, Noun Phrase Chunking and Named Entities. 

As mentioned in the Text Cleaning section, these are powerful in combination and not in isolation. We will learn how to use pipelines in spaCy for the same. 

We will start with a walkthrough of the official spaCy guidelines and code examples. This uses the pre-trained models from spaCy itself. 

## Linguistics

### Linguistics Application: Chatbots
Chat bots or conversational systems such as Siri need to have intricate understanding of language to do two main things: 
1. Understand human input in either text or voice
    - this input is different from how we use search, for instance we might enter the exact item we want to buy in Amazon search but we will might Alexa for suggestions on best toys for 3 year olds

2. Generate language response
    - What does a Google search for *Steve Jobs Birthday* return for you? A list of web pages. On the other hand, you would expect Siri not only tell the exact date of Job's birth - but also a proper sentence such as: *Steve Jobs was born on 24 Feb 1955*


The way we study language is referred to as linguistics. This section covers language concepts as applied to Natural Language Processing. 

We have seen some of this when we studied English Grammar back in our school days. Famously, you might want to recap the following: _noun_, _verb_, _gerund_ and so on. 


## Main Headings :
- HEADING 1: Linguistic Roots of English Language
- HEADING 2: Leveraging Language: Example Tasks - Chat bots and Search
- HEADING 3: PoS Tagging, NP Chunking
- HEADING 4: NER: Inbuilt models
- HEADING 5: Gluing it all together

## Skills learned:
- SKILL 1: Linguistic Concepts in NLP
- SKILL 2: Spelling Correction, Slot Filling
- SKILL 3: Linguistic Tasks using spaCy
- SKILL 4: NER with spaCy pipelines
- SKILL 5: End to end spacy implementation with a toy example

### Slot Filling vs NER
The goals of slot filling are different. Slot filling is looking for specific pieces of information with respect to something. For instance, you might ask Siri or Google Assistant - _Who is the spouse of Sachin Tendulkar?_

In this example, you are looking for _spouse_ with respect to _Sachin Tendulkar_. 

Now this response information can be named entities _e.g. who is spouse of this person?_ but can also be other things _e.g. when was this person born?_ 

Exactly what information depends on the application, but Wikipedia info boxes are a good example. 
TKX Add Wiki info box screenshot of Sachin Tendulkar
TKX TIP: Similar thought process is quite useful for building a chatbot, say for customer service. 

NER is more generic and just looks for things. When we mean things, usually they are nouns such as names, like people, companies, places, etc. Your focus is not on the relation between these things. 

For instance, BookMyShow allows you to book tickets via WhatsApp. 

TKX: Add BMS screenshots from phone

TKX: This example needs to be worked out better
The relation is a movie, which has following _slots_:date, screen, theatre/movie hall name. A NER system would just tag <Bengaluru> and <movie name> as names of things, as opposed to 'dfsdf,' which is not the name of a specific individual thing. If the sentence said 'adasda' instead of 'adasda,' NER would pick that out as a movie name. 
    
Some NER taggers will also extract dates, money, and other numbers because they're useful, even though they're not really named entities.

These tasks don't HAVE to be done using sequence tagging, either; slot-filling has been done with templates and multi-stage approaches (extract candidate phrases by tagging and then classify or rank).
