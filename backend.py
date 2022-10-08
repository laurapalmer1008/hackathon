import datetime
import json
import re

import flask
import pickle
import gensim as gensim
import pandas
import pandas as pd
import requests
from gensim import corpora
import pymorphy2

def get_data():
    mask = "https://www.kommersant.ru/archive/rubric/3/month/{:s}"
    month = str(datetime.date.today())
    initial_url = mask.format(month)
    r = requests.get(initial_url)
    articles = re.findall(r"data-article-url=\"[^\"]*", r.text)
    last_article = articles[-1][articles[-1].rfind("/") + 1:]
    data_dump = []
    try:
        new_url = "https://www.kommersant.ru/listpage/lazyloaddocs?regionid=77&listtypeid=1&listid=4&date={:s}&intervaltype=3&idafter=".format(
            month[8:] + "." + month[5:7] + "." + month[:4]) + str(last_article)
        r = requests.get(new_url)
        data = r.json()
        for item in data['Items']:
            last_article = item['DocsID']
            article_url = "https://www.kommersant.ru/doc/" + str(last_article)
            title = item['Title']
            subtitle = item['SubTitle']
            tags = item['Tags']
            article_response = requests.get(article_url, timeout=10)
            views = int(re.findall(r"data-article-views=\"\d*\"", article_response.text)[0][20:-1])
            article_date = re.findall(r"data-article-daterfc822=\"[^\"]*\"", article_response.text)[0][25:-1]
            texts = re.findall(r"<p class=\"doc__[^>]*>.*</p>", article_response.text)
            article_texts = []
            for t in texts:
                article_texts.append(re.sub(r"<[^>]*>", "", t))
            obj = {"article_id": last_article, "article_url": article_url, "title": title, "subtitle": subtitle,
                   "tags": tags, "article_texts": article_texts, "views": views, "datetime": article_date}
            data_dump.append(obj)
        return data_dump
    except:
        return None


def Function_make_corp(df):
    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    def remove_stopwords(texts):
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatize(doc):
        # doc = re.sub(patterns, ' ', doc)
        tokens = []
        for token in doc:
            token = str(token)
            token = morph.normal_forms(token)

            tokens.append(token)
        if len(tokens) > 2:
            return tokens
        return None

    data = [df]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    data_words = list(sent_to_words(data))

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    morph = pymorphy2.MorphAnalyzer()

    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops)
    data_words_bigrams = data_words_bigrams[0]
    data_lemmatized = lemmatize(data_words_bigrams)

    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]

    return corpus


def get_result(data_corpus, id):
    for p in model.get_document_topics(data_corpus):
        for i in p:
            if i[0] in id and i[1] > 0.15:
                return 1
    return 0


app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False
stop_words = json.load(open("stop_words.json", "r", encoding='utf8'))
with open('Model_komm.pickle', 'rb') as f:
    model = pickle.load(f)
with open('id2word.pickle', 'rb') as fi:
    id2word = pickle.load(fi)


@app.route("/accountant_news")
def get_accountant_news():
    data = pd.DataFrame(get_data())
    role = []
    for elem, news in enumerate(data.article_texts):
        processed_news = " ".join(news)
        processed_news = Function_make_corp(processed_news)
        result = get_result(processed_news, [1, 6, 7])
        if result == 1:
            role.append(elem)
    return [" ".join(news) for news in data.article_texts.iloc[role].values.tolist()][:3]


@app.route("/director_news")
def get_director_news():
    data = pd.DataFrame(get_data())
    role = []
    for elem, news in enumerate(data.article_texts):
        processed_news = " ".join(news)
        processed_news = Function_make_corp(processed_news)
        result = get_result(processed_news, [2, 8, 11, 15])
        if result == 1:
            role.append(elem)
    return [" ".join(news) for news in data.article_texts.iloc[role].values.tolist()][:3]
