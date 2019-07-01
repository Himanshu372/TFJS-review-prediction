from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory
import json
# import cv2
import pandas as pd
import numpy as np
import io
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
use_stemmer=True
porter_stemmer=PorterStemmer()
stop_words=set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')

@app.route("/api/prepare", methods=["POST"])
def prepare():
    str = request.form['formInput1']
    res = preprocessing(str)
    res = json.dumps({"ndarray": res.tolist()})
    print('res' , res)
    return res

@app.route('/model')
def model():
    json_data = json.load(open("./model_json/model.json"))
    return jsonify(json_data)


@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('model_json', path)







def preprocessing(s):
    '''

    :param s:
    :return:
    '''
    # embeddings_index = read_glove('./Embeddings/glove.6B.100d.txt')
    reviews = pd.read_csv('./data/processed_review_text_en.csv')
    reviews['text_data'] = reviews['text_data'].apply(lambda x : str(x))
    combined_docs = reviews['text_data'].values
    # print(type(combined_docs))
    t = Tokenizer()
    t.fit_on_texts(texts=combined_docs)
    tokenizer_json = t.to_json()
    with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    print(s)
    processed_str = preprocess_desc(s)
    print(processed_str)
    encoded_docs = t.texts_to_sequences([processed_str])
    max_length = 40
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # file.save("static/UPLOAD/img.png") # saving uploaded img
    # cv2.imwrite("static/UPLOAD/test.png", res) # saving processed image
    print(padded_docs.shape)
    return padded_docs


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;%')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)

    if word in stop_words:
        return ''
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def preprocess_desc(desc, use_stemmer=True):
    # try:
    processed_desc = []
    # Convert to lower case
    desc = desc.lower()

    # Remove RT (redesc)
    desc = re.sub(r'\brt\b', '', desc)
    # Replace 2+ dots with space
    desc = re.sub(r'\.{2,}', ' ', desc)
    # Strip space, " and ' from desc
    desc = desc.strip(' "\'')
    # Replace multiple spaces with a single space
    desc = re.sub(r'\s+', ' ', desc)
    # remove html tags
    desc = re.sub('<.*?>', '', desc)

    words = desc.split()
    #     return words
    # remove stop words
    #     words=[w for w in words if w not in stop_words]
    #     return words
    print(words)
    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word = str(porter_stemmer.stem(word))
            #                 word=str(wordnet_lemmatizer.lemmatize(word))
            processed_desc.append(word)

    return ' '.join(processed_desc)
    # except:
    #     print('Issue with desc', desc)


if __name__ == "__main__":
    app.run()