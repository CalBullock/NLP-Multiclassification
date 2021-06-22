#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
get_ipython().system('pip install bert-for-tf2')
get_ipython().system('pip install tensorflow_hub')
import tensorflow_hub as hub
from bert import bert_tokenization
import tensorflow as tf
from tensorflow import keras
import zipfile


'''Function to apply predictions to input excel file and return the output excel file'''
def excel_read_write_predictions(input_file, model):
    
    '''Read the input excel file'''
    pred_df = pd.read_excel(input_file) 

    '''Setup requrements for BERT'''
    module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
    bert_layer = hub.KerasLayer(module_url, trainable=True)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

    '''Function to create the embeddings'''
    def bert_encode(texts, tokenizer, max_len=512):
        all_tokens = []
        all_masks = []
        all_segments = []

        for text in texts:
            text = tokenizer.tokenize(text)

            text = text[:max_len-2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_len - len(input_sequence)

            tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_len

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
    max_len = 150
    temp_vector = bert_encode(pred_df['review'], tokenizer, max_len=max_len)
  
    '''Predict the reviews request'''
    temp_proba = model.predict(temp_vector)
    
    '''Add each probability to the dataframe'''
    pred_df['prob_l+'] = temp_proba[:,0]
    pred_df['prob_l-'] = temp_proba[:,1]
    pred_df['prob_rl'] = temp_proba[:,2]
    pred_df['prob_yl'] = temp_proba[:,3]
    
  
    '''Print the output'''
    print(pred_df)

    '''Export the dataframe to excel with the title'''
    return pred_df.to_excel(excel_writer='reviews_with_predictions.xlsx', index=False)


'''Unzip model that was saved'''
with zipfile.ZipFile('bert_full_model.zip', 'r') as zip_ref:
    zip_ref.extractall('bert_full_model_unzip')

'''Load the unzipped model'''
model = keras.models.load_model('bert_full_model_unzip/content/bert_full_model')

'''Create a test dataframe'''
test_df = pd.DataFrame()
test_df['id'] = ['1234', '2345', '3456']
test_df['review'] = ['good but it needs to be in korean', 'poorly translated', 'they did a good job with the translation']

'''Convert dataframe to excel to run in function'''
test_df.to_excel(excel_writer='test_excel.xlsx', index=False)

pd.read_excel('test_excel.xlsx')

'''Test the function'''
excel_read_write_predictions('test_excel.xlsx', model)

'''Print output of funcion'''
pd.read_excel('reviews_with_predictions.xlsx')
