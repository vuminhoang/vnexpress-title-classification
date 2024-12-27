import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

# -------------------------GUIDE--------------------------------------------------------
'''
Run these commands on terminal:
pip install huggingface-hub
huggingface-cli login
--> after that go to your hfhub and get the access token
'''

# # define model and trained weights
# checkpoint = 'distilbert-base-multilingual-cased'
# model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=8)
# model.load_weights('model/title_classification_weights.h5')
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# # create a repo on huggingface hub, and push your model
# model.push_to_hub("minnehwg/vnexpress-title-classification")

# ---------------------------------------------------------------------------------------
# CHECK IT OUT!

# label matching function
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

# define checkpoint, model, tokenizer
checkpoint = 'minnehwg/vnexpress-title-classification'
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# predict function
def predict_sentence(model, tokenizer, sentence):
    input_data = tokenizer(sentence, return_tensors='tf', padding=True, truncation=True)
    logits = model(input_data['input_ids'], attention_mask=input_data['attention_mask']).logits
    probabilities = tf.nn.softmax(logits, axis=1)
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    highest_probability = probabilities.numpy()[0, predicted_class]
    title = convert_label_to_title(predicted_class)
    return title, probabilities.numpy(), highest_probability

# -----------------TEST---------------------------------------------------------------------------
sentence  = "Trấn Thành là MC số 1 của Vieon"
title, probabilities, highest = predict_sentence(model, tokenizer, sentence)
print(f"Tiêu đề cần dự đoán: {sentence}")
print(f"Danh mục dự đoán: {title}")
print(f"Xác suất dự đoán: {highest}")

# Tiêu đề cần dự đoán: Trấn Thành là MC số 1 của Vieon
# Danh mục dự đoán: GIẢI TRÍ
# Xác suất dự đoán: 0.8793759346008301
