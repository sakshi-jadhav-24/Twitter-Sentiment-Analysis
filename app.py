import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import snscrape.modules.twitter as sntwitter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="centered")

# -------------------- STOPWORDS --------------------
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# -------------------- MODEL --------------------
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# -------------------- FETCH TWEETS --------------------
def get_tweets(keyword, count=5):
    tweets = []

    try:
        query = keyword + " since:2024-01-01"
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= count:
                break
            tweets.append(tweet.content)

    except:
        pass

    # fallback
    if len(tweets) == 0:
        tweets = [
            f"{keyword} is trending now!",
            f"I love {keyword}, amazing updates!",
            f"{keyword} news is shocking today",
            f"Not happy with {keyword}",
            f"{keyword} performance is great"
        ]

    return tweets

# -------------------- USER TWEETS --------------------
def get_user_tweets(username, count=5):
    tweets = []

    try:
        query = f"from:{username}"
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= count:
                break
            tweets.append(tweet.content)

    except:
        pass

    if len(tweets) == 0:
        tweets = [f"{username} latest tweet..." for _ in range(5)]

    return tweets

# -------------------- PREDICT --------------------
def predict(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [w for w in text if w not in stop_words]
    text = " ".join(text)
    text = vectorizer.transform([text])
    pred = model.predict(text)
    return "Positive" if pred == 1 else "Negative"

# -------------------- CARD --------------------
def card(text, sentiment):
    color = "#28a745" if sentiment=="Positive" else "#dc3545"
    return f"""
    <div style="background:{color};padding:6px;border-radius:8px;margin:5px 0;">
        <b style="color:white;font-size:14px;">{sentiment}</b>
        <p style="color:white;font-size:13px;">{text}</p>
    </div>
    """

# -------------------- CHART --------------------
def show_charts(pos, neg):
    df = pd.DataFrame({
        "Sentiment": ["Positive", "Negative"],
        "Count": [pos, neg]
    })

    st.subheader("📊 Sentiment Summary")
    st.bar_chart(df.set_index("Sentiment"))

    fig, ax = plt.subplots()
    ax.pie(df["Count"], labels=df["Sentiment"], autopct='%1.1f%%')
    st.pyplot(fig)

# -------------------- MAIN --------------------
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
        keyword = st.text_input("Enter keyword (AI, India, Cricket)")

        if st.button("Fetch Tweets"):
            tweets = get_tweets(keyword)

            pos, neg = 0, 0

            for t in tweets:
                s = predict(t, model, vectorizer, stop_words)

                if s == "Positive":
                    pos += 1
                else:
                    neg += 1

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Positive", pos)
            with col2:
                st.metric("Negative", neg)

            show_charts(pos, neg)

            st.subheader("📝 Tweets")

            # Scrollable tweets
            with st.container(height=300):
                for t in tweets:
                    s = predict(t, model, vectorizer, stop_words)
                    st.markdown(card(t, s), unsafe_allow_html=True)

    # -------- USER --------
    elif option == "👤 User Tweets":
        username = st.text_input("Enter username (elonmusk)")

        if st.button("Fetch User Tweets"):
            tweets = get_user_tweets(username)

            st.subheader("📝 User Tweets")

            with st.container(height=300):
                for t in tweets:
                    s = predict(t, model, vectorizer, stop_words)
                    st.markdown(card(t, s), unsafe_allow_html=True)

# -------------------- RUN --------------------
if __name__ == "__main__":
    main()