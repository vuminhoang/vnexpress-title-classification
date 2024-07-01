import streamlit as st
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np

def convert_label_to_title(label):
  convert_dict = {
    0: "SỨC KHỎE",
    1: "GIÁO DỤC",
    2: "THỂ THAO",
    3: "PHÁP LUẬT",
    4: "KHOA HỌC",
    5: "DU LỊCH",
    6: "GIẢI TRÍ",
    7: "KINH DOANH"
  }
  return convert_dict[label]

def predict_sentence(model, tokenizer, sentence):
    input_data = tokenizer(sentence, return_tensors='tf', padding=True, truncation=True)
    logits = model(input_data['input_ids'], attention_mask=input_data['attention_mask']).logits
    probabilities = tf.nn.softmax(logits, axis=1)
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    highest_probability = probabilities.numpy()[0, predicted_class]
    title = convert_label_to_title(predicted_class)
    return title, probabilities.numpy(), highest_probability

@st.cache_resource
def load_model(checkpoint, num_class):
  model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_class)
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
  return model, tokenizer

checkpoint = 'distilbert-base-multilingual-cased'
model, tokenizer = load_model(checkpoint, 8)
model.load_weights('best_model_weights.h5')

text = st.text_area('Nhập tiêu đề vào đây')

if text:
    title, probabilities, highest = predict_sentence(model, tokenizer, text)
    out = {
        'title': title,
        'prob': highest
    }
    st.json(out)
