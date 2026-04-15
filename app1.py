import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# -------------------- TOKEN --------------------
BEARER_TOKEN = "PASTE_YOUR_API_KEY"

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

# -------------------- TWITTER CLIENT --------------------
@st.cache_resource
def get_client():
    return tweepy.Client(bearer_token=BEARER_TOKEN)

# -------------------- SEARCH TWEETS --------------------
def get_tweets(keyword, count=10):
    try:
        client = get_client()

        response = client.search_recent_tweets(
            query=keyword + " -is:retweet lang:en",
            max_results=min(count, 10),
            tweet_fields=["text"]
        )

        if response.data is None:
            return []

        return [tweet.text for tweet in response.data]

    except Exception as e:
        st.error(f"API Error: {e}")
        return []

# -------------------- USER TWEETS --------------------
def get_user_tweets(username, count=5):
    try:
        client = get_client()

        # username -> user id
        user = client.get_user(username=username)

        if user.data is None:
            return []

        user_id = user.data.id

        tweets = client.get_users_tweets(
            id=user_id,
            max_results=min(count, 10),
            tweet_fields=["text"]
        )

        if tweets.data is None:
            return []

        return [tweet.text for tweet in tweets.data]

    except Exception as e:
        st.error(f"User Fetch Error: {e}")
        return []

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
    <div style="background:{color};padding:10px;border-radius:10px;margin:10px 0;">
        <b style="color:white;">{sentiment}</b>
        <p style="color:white;">{text}</p>
    </div>
    """

# -------------------- CHART --------------------
def show_charts(pos, neg):
    df = pd.DataFrame({
        "Sentiment": ["Positive", "Negative"],
        "Count": [pos, neg]
    })

    st.bar_chart(df.set_index("Sentiment"))

    fig, ax = plt.subplots()
    ax.pie(df["Count"], labels=df["Sentiment"], autopct='%1.1f%%')
    st.pyplot(fig)

# -------------------- MAIN --------------------
def main():
    st.set_page_config(page_title="Sentiment App", layout="wide")
    st.title("🔥 Real-Time Twitter Sentiment Dashboard")

    stop_words = load_stopwords()
    model, vectorizer = load_model()

    option = st.selectbox("Select Mode", [
        "Manual Text",
        "🔥 Keyword Analysis",
        "👤 User Tweets Analysis"
    ])

    # -------- MANUAL --------
    if option == "Manual Text":
        text = st.text_area("Enter text")
        if st.button("Analyze"):
            res = predict(text, model, vectorizer, stop_words)
            st.success(f"Sentiment: {res}")

    # -------- KEYWORD --------
    elif option == "🔥 Keyword Analysis":
        keyword = st.text_input("Enter keyword (AI, India, Cricket)")

        if st.button("Fetch Tweets"):
            tweets = get_tweets(keyword)

            if len(tweets) == 0:
                st.warning("No tweets found (try different keyword)")
                return

            pos, neg = 0, 0

            for text in tweets:
                s = predict(text, model, vectorizer, stop_words)

                if s == "Positive":
                    pos += 1
                else:
                    neg += 1

                st.markdown(card(text, s), unsafe_allow_html=True)

            st.metric("Positive", pos)
            st.metric("Negative", neg)

            show_charts(pos, neg)

    # -------- USER --------
    elif option == "👤 User Tweets Analysis":
        username = st.text_input("Enter username (elonmusk, nasa)")

        if st.button("Fetch User Tweets"):
            tweets = get_user_tweets(username)

            if len(tweets) == 0:
                st.warning("No tweets found or invalid username")
                return

            for text in tweets:
                s = predict(text, model, vectorizer, stop_words)
                st.markdown(card(text, s), unsafe_allow_html=True)

# -------------------- RUN --------------------
if __name__ == "__main__":
    main()