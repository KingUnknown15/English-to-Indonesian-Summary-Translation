import numpy as np
import pandas as pd
import pickle
from tensorflow import keras 
from keras.preprocessing.text import Tokenizer 
from keras.utils import pad_sequences
from keras.models import Model
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Load Summary Model and Tokenizer
sum_encoder = keras.models.load_model('SummaryEncoder')
sum_decoder = keras.models.load_model('SummaryDecoder')
max_news = 88
max_sum = 34

with open('sumdatatokenizer.pickle','rb') as handle:
    data_sum_tokenizer = pickle.load(handle)

with open('sumlabeltokenizer.pickle','rb') as handle:
    label_sum_tokenizer = pickle.load(handle)

#Load Translation Model and Tokenizer
trans_encoder = keras.models.load_model('TransEncoder')
trans_decoder = keras.models.load_model('TransDecoder')
max_eng = 35
max_idn = 35

with open('transdatatokenizer.pickle','rb') as handle:
    data_trans_tokenizer = pickle.load(handle)

with open('translabeltokenizer.pickle','rb') as handle:
    label_trans_tokenizer = pickle.load(handle)


rev_news_word_index = data_sum_tokenizer.index_word
rev_summary_word_index = label_sum_tokenizer.index_word
summary_word_index = label_sum_tokenizer.word_index

rev_english_word_index = data_trans_tokenizer.index_word
rev_indonesia_word_index = label_trans_tokenizer.index_word
indonesia_word_index = label_trans_tokenizer.word_index

def generate(input, encoder_model, decoder_model, data_tokenizer, max_data, max_label,label_word_index,rev_label_word_index):
  input_seq = data_tokenizer.texts_to_sequences([input])
  input_seq = pad_sequences(input_seq,maxlen=max_data,padding='post')
  input_seq = input_seq.reshape(1, max_data)
  print(type(input_seq))
  e_out, e_h, e_c = encoder_model.predict(input_seq)
  
  target_seq = np.zeros((1, 1))
  target_seq[0, 0] = label_word_index['hajime']
  print(target_seq)
  
  summary = ''
  while True:
    output_tokens, h, c = decoder_model.predict([target_seq]+ [e_out, e_h, e_c])
    idx = np.argmax(output_tokens[0, -1, :])
    sample_word = rev_label_word_index[idx]
    #print(sample_word)
    if (sample_word != 'owari'):
      summary += ' '+sample_word
    if (sample_word == 'owari' or len(summary.split())>=(max_label-1)):
      break
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = idx
    e_h, e_c = h,c

  return summary
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


app = Flask(__name__)
api = Api(app)

class SummTrans(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('text')
        args = parser.parse_args()
        try:
           summary = generate(args.text,sum_encoder,sum_decoder,data_sum_tokenizer,max_news,max_sum,summary_word_index,rev_summary_word_index)
           translate = generate(summary,trans_encoder,trans_decoder,data_trans_tokenizer,max_eng,max_idn,indonesia_word_index,rev_indonesia_word_index)
           return jsonify({'summary':summary,'final':translate})
        except Exception as e:
           return jsonify({'error':str(e)})

api.add_resource(SummTrans,'/api/summtrans')

if __name__ == '__main__':
    app.run()