## VnExpress Title Classification
This repository was created to classify headlines from VnExpress, a famous newspaper in Vietnam, into 8 different labels, including: 
- "Sức Khỏe" (Health)
- "Giáo Dục" (Education)
- "Thể Thao" (Sport)
- "Pháp Luật" (Law)
- "Khoa Học" (Science)
- "Du Lịch" (Travel)
- "Giải Trí" (Entertainment)
- "Kinh Doanh" (Business)

## Newest update
- a demo on google colab with pyngrok and flask, check it out!. Date: 23/1/2025.

## Data Collection
- Crawl on the Vnexpress website.
- Or u can use my data in raw_data folder.

This dataset was kindly provided by one of my teachers at HUS, Mr. Nguyen Tuan Anh. I am deeply grateful for his generosity and support, which allowed me to use this dataset as a starting point for training my first NLP models. Thank you, Mr. Nguyen Tuan Anh, for your invaluable contribution to my learning journey!

## Pre-processing & training
- Described in title_cls_train.ipynb file.

## Demo
Check out my demo at: https://huggingface.co/spaces/minnehwg/vnexpress-title-classification (seems like it died, i'll create a new version soon!)

```bash
git clone https://github.com/vuminhoang/vnexpress-title-classification.git
cd vnexpress-title-classification
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Near-future Plans
- [x] Create demo on colab with pyngrok + flask!
- [ ] Create a pytorch version.
- [ ] Convert it to onnx, compare the speed and accuracy between the 2 versions.
- [ ] Build Flask App.
- [ ] Build a dockerfile: the main purpose is to learn how to use docker to package product.


  
