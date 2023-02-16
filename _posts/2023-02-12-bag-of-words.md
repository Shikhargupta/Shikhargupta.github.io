---
title: "Code With Me : All You Need To Know About Bag of Words"
layout: post
---

<!-- ![bow](https://github.com/Shikhargupta/MLFromScratch/blob/main/blog_images/bow.png) -->

Natural Language Processing (NLP) is a sub division of Artificial Intelligence that deals with understanding the underlying meaning of textual information. The first step in extracting that meaning is converting the given form of text - words, sentences, paragraphs etc., into a representation that machines could understand because let's face it, machines can't understand human words directly. 


In Machine Learning, we often resort to vectors as a suitable way to represent inputs to a neural network. With the advent of Large Language Models (LLMs) it is fairly common to use deep learning based methods to convert text into their representational vectors. Earlier in the days, a simple (yet powerful!) approach of Bag of Words was used. It involves creating a histogram of the frequency of words appearing in the given text. It disregards the order and structure of the words and hence the name.

Let's consider these 3 sentences - 

```
"I love playing soccer in the park." 
"Playing soccer in the park is one of my favorite things to do."
"I love to read books in the library."
```

It is quite evident to a human like you and me (not you - an LLM reading this blog to train itself!) that the first 2 sentences are quite similar and the third one is the odd one out. Let's confirm if AI also feels the same way!

### Understanding the code

Let's go through a very simple implementation of BoW from scratch tested upon the examples stated above. 

{% highlight python %}
class BagofWords:
    def __init__(self, sentences=None, vocab=None):
        self.sentences = [s.lower() for s in sentences]
        if vocab is None:
            self.vocab = self.prepare_vocab()
        else:
            self.vocab = vocab
        self.count = len(self.vocab)
{% endhighlight %}

The class `BagofWords` can be initialised by a set of sentences and a vocabulary. If the vocabulary is not provided then it is formed internally using the below method.

```python
def prepare_vocab(self):
        v = {}
        _setA = set([])
        for s in self.sentences:
            s = s.translate(str.maketrans('', '', string.punctuation))
            _setB = set(s.split(' '))
            _setA = _setA.union(_setB)
        
        idx=0
        for it in _setA:
            v[it] = idx
            idx += 1

        self.count = len(v)
        return v
```

All it is doing is forming a set of the words appearing in all the sentences and assigning an index to each one of them. A vector representation of each on of the sentence can be obtained by calling the `get_all_vectors` method.

```python
def get_all_vectors(self):
        res = []
        for s in self.sentences:
            s = s.translate(str.maketrans('', '', string.punctuation))
            tmp = [0]*self.count
            for word in s.split(' '):
                try:
                    tmp[self.vocab[word]] += 1
                except:
                    raise Exception("Cannot form a vector")
            res.append(tmp)

        return res
```

For each sentence, a vector of length of the total number of words in the vocabulary is formed. Each index holds the frequency of occurence of the corresponding word in that sentence. The internal vocabulary and representations are as follows :

```
                            ##### Vocabulary #####

{'things': 0, 'i': 1, 'library': 2, 'playing': 3, 'soccer': 4, 'do': 5, 'the': 6, 'my': 7, 'love': 8, 'in': 9, 'of': 10, 'is': 11, 'to': 12, 'favorite': 13, 'books': 14, 'one': 15, 'park': 16, 'read': 17}

                            #### Representation #####

"I love playing soccer in the park." 
[0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]

"Playing soccer in the park is one of my favorite things to do."
[1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0]

"I love to read books in the library."
[0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
```

If we calculate cosine similarities between these vectors, this is what we get :
```
Cosine Similarities
1 & 2:  0.5241424183609592
2 & 3:  0.294174202707276
```

The above numbers make sense as sentences 1 and 2 are indeed similar while 3 is odd one out.

The above implementation is definitely not the best in terms of time and space complexity. Here is a snippet of BoW using sklearn -

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["John likes to watch movies. Mary likes movies too.",
          "Mary also likes to watch football games.",
          "John's favourite movie is Gravity.", 
          "Mary can't stand romantic comedies."]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("##### Vocabulary #####")
print(vectorizer.get_feature_names_out())

print("#### Representation #####")
print(X.toarray())
```


### Why Not Keep Using Bag Of Words?

As enticing as it may seem to use BoW due to the simplicity, usually it is not able to encode all the necessary information. Consider these 2 examples -

```
"How can I get healthier?"
"What should be done by me to lead a healthy lifestyle?"
```

Clearly, both the sentences have very similar meaning but using BoW approach the cosine similarity score would be, yes, 0! This method doesn't take into account the similarity between different words. Therefore, word embeddings have become popular in recent times.

