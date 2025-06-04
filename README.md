# 🧠 SentimentScope AI

SentimentScope AI is an advanced NLP-powered web app built with **Streamlit** and **TensorFlow** that detects **18 different emotions and sentiments** from user-provided text. It uses a deep learning model trained on over 1M labeled samples to provide nuanced emotional insights.

![SentimentScope AI Screenshot](demo_screenshot.png) <!-- Optional: Add a screenshot image here -->

---

## 🚀 Features

- Detects 18 sentiment/emotion types: Positive, Negative, Joy, Sadness, Fear, Love, Trust, etc.
- Beautiful Streamlit interface with custom CSS
- Emoji-enhanced output and interactive charts
- Example inputs and real-time confidence visualizations
- Full text preprocessing and emotion tagging
- Pretrained model and tokenizer included

---

## 🧱 Tech Stack

- **Frontend**: Streamlit + Custom HTML/CSS
- **Backend**: TensorFlow (Keras), NLTK, Scikit-learn
- **Model**: Transformer-based sentiment classifier
- **Data**: Trained on 1M+ labeled emotional text samples

---
## Technologies Used

- **Streamlit**: For building the web application.
- **TensorFlow/Keras**: For the machine learning model.
- **NLTK**: For natural language processing tasks like tokenization, stemming, and stopword removal.
- **Pickle**: For loading pre-trained models and tokenizers.
## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tarekrahamn/sentimentscope-ai.git
   cd sentimentscope-ai
2. Install the required libraries:
   ```bash
   pip install streamlit tensorflow nltk
   ```

3. Download necessary NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. Place your `model.pkl` and `token.pkl` files in the same directory as the app.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the application in your web browser.
2. Enter the tweet you want to analyze in the input field.
3. Click the "Predict" button to see the sentiment classification.

## Model Details

The sentiment analysis model has been trained on a dataset of tweets. It preprocesses the input text by:

- Converting text to lowercase
- Removing punctuation and emojis
- Stemming and removing stopwords


The model outputs a probability score indicating the sentiment, where a score of 0.5 or higher is classified as positive.

## Contributing

If you'd like to contribute to this project, feel free to open an issue or submit a pull request.
## Acknowledgments

- [Streamlit](https://streamlit.io/) - For creating the web app framework.
- [TensorFlow](https://www.tensorflow.org/) - For the deep learning model.
- [NLTK](https://www.nltk.org/) - For natural language processing tools.

