import streamlit as st
import pickle
import re
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

st.set_page_config(page_title="Sentiment Dashboard", layout="centered")

# ---------------- STOPWORDS ----------------
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# ---------------- FAKE REAL-TIME DATA ----------------
def generate_tweets(keyword, count=5):
    templates = [
        f"{keyword} is amazing!",
        f"I love {keyword} so much",
        f"{keyword} is not good",
        f"{keyword} trending worldwide",
        f"People are talking about {keyword}",
        f"{keyword} is disappointing",
        f"Best experience with {keyword}",
        f"Worst update about {keyword}"
    ]
    return random.sample(templates, count)

# ---------------- USER DATA ----------------
def generate_user_tweets(username, count=5):
    return [f"{username} latest tweet {i}" for i in range(1, count+1)]

# ---------------- PREDICT ----------------
def predict(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [w for w in text if w not in stop_words]
    text = " ".join(text)
    text = vectorizer.transform([text])
    pred = model.predict(text)
    return "Positive" if pred == 1 else "Negative"

# ---------------- CARD ----------------
def card(text, sentiment):
    color = "#28a745" if sentiment=="Positive" else "#dc3545"
    return f"""
    <div style="background:{color};padding:6px;border-radius:8px;margin:5px 0;">
        <b style="color:white;">{sentiment}</b>
        <p style="color:white;">{text}</p>
    </div>
    """

# ---------------- CHART ----------------
def show_charts(pos, neg):
    df = pd.DataFrame({
        "Sentiment": ["Positive", "Negative"],
        "Count": [pos, neg]
    })

    st.bar_chart(df.set_index("Sentiment"))

    fig, ax = plt.subplots()
    ax.pie(df["Count"], labels=df["Sentiment"], autopct='%1.1f%%')
    st.pyplot(fig)

# ---------------- MAIN ----------------
def main():
    st.title("🔥 Twitter Sentiment Dashboard")

    stop_words = load_stopwords()
    model, vectorizer = load_model()

    option = st.selectbox("Select Mode", [
        "Manual Text",
        "🔥 Keyword Analysis",
        "👤 User Tweets"
    ])

    # -------- MANUAL --------
    if option == "Manual Text":
        text = st.text_area("Enter text")
        if st.button("Analyze"):
            result = predict(text, model, vectorizer, stop_words)
            st.success(f"Sentiment: {result}")

    # -------- KEYWORD --------
    elif option == "🔥 Keyword Analysis":
        keyword = st.text_input("Enter keyword")

        if st.button("Fetch Tweets"):
            tweets = generate_tweets(keyword)

            pos, neg = 0, 0

            for t in tweets:
                s = predict(t, model, vectorizer, stop_words)
                if s == "Positive":
                    pos += 1
                else:
                    neg += 1

            col1, col2 = st.columns(2)
            col1.metric("Positive", pos)
            col2.metric("Negative", neg)

            show_charts(pos, neg)

            with st.container(height=300):
                for t in tweets:
                    s = predict(t, model, vectorizer, stop_words)
                    st.markdown(card(t, s), unsafe_allow_html=True)

    # -------- USER --------
    elif option == "👤 User Tweets":
        username = st.text_input("Enter username")

        if st.button("Fetch User Tweets"):
            tweets = generate_user_tweets(username)

            with st.container(height=300):
                for t in tweets:
                    s = predict(t, model, vectorizer, stop_words)
                    st.markdown(card(t, s), unsafe_allow_html=True)

if __name__ == "__main__":
    main()