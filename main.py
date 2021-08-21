import json
from itertools import combinations
import nltk
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.stem.snowball import RussianStemmer
import networkx as nx
import sys

def similarity(s1, s2):
    if not len(s1) or not len(s2):
        return 0.0
    return len(s1.intersection(s2))/(1.0 * (len(s1) + len(s2)))

def textrank(text):
    sentences = sent_tokenize(text)
    tokenizer = RegexpTokenizer(r'\w+')
    lmtzr = RussianStemmer()
    words = [set(lmtzr.stem(word) for word in tokenizer.tokenize(sentence.lower()))
             for sentence in sentences] 	 
    pairs = combinations(range(len(sentences)), 2)
    scores = [(i, j, similarity(words[i], words[j])) for i, j in pairs]
    scores = filter(lambda x: x[2], scores)
    g = nx.Graph()
    g.add_weighted_edges_from(scores)
    pr = nx.pagerank(g)
    return sorted(((i, pr[i], s) for i, s in enumerate(sentences) if i in pr), key=lambda x: pr[x[0]], reverse=True)

def sumextract(text, n=5):
    tr = textrank(text)
    top_n = sorted(tr[:n])
    return ' '.join(x[2] for x in top_n)

def load_json(json_path):
    return json.load(open(json_path))

def analysis_news(news_one):
  if news_one['headline'] is not None:
    news = news_one['headline']
  else:
    news = news_one['body']
  news_one = news.replace('/', '')
  news_one = news_one.replace('\\', '')
  news_one = news_one.replace('ар+2', '')
  news_one = news_one.replace('2ар+1', '')
  news_one = news_one.replace('ар++', '')
  news_one = news_one.replace('мк+1', '')
  news_one = news_one.replace('ak+3', '')
  news_one = news_one.replace('of+3', '')
  news_one = news_one.lower()
  return news_one


def sorted_all_dict(my_d):
  return sorted(my_d.items(), key=lambda x: x[1], reverse=True)


send_data = []
for stack_news in load_json(sys.argv[1]):
  word_one = []
  for one_news in stack_news['news']:
    word_not_one = analysis_news(one_news)
    if word_not_one in word_one:
      #nothing
      pass
    else:
      word_one.append(word_not_one)
  send_data.append('. '.join(word_one))


for text in send_data:
  target = sumextract(text, 1)

  # final_target = target.split(',')[0]

  print(target)






