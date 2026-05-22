import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Mood Music Recommender",
    page_icon="🎧",
    layout="centered"
)

st.title("🎧 Mood-Based Music Recommender")
st.markdown("Tell me how you're feeling, and I'll suggest the perfect vibe 🎶")


@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except Exception as e:
        return None, str(e)

model, vectorizer = load_model()

if model is None:
    st.error(f"🚨 Error loading model: {vectorizer}")
    st.stop()

user_input = st.text_input("💭 How are you feeling today?")

if st.button("Recommend Music"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter how you're feeling first!")
    else:
        
        vec = vectorizer.transform([user_input])
        emotion = model.predict(vec)[0]

        st.success(f" Detected Mood: **{emotion}**")

        if emotion == "Happy":
            st.markdown("🎶 **Upbeat vibes coming your way!**")
            st.write("- Party songs")
            st.write("- Dance tracks")
            st.write("- Pop hits")

        elif emotion == "Sad":
            st.markdown("🎧 **Take it slow, feel the music**")
            st.write("- Soft acoustic songs")
            st.write("- Emotional ballads")
            st.write("- Lo-fi beats")

        elif emotion == "Angry":
            st.markdown("🔥 **Let it out with energy!**")
            st.write("- Rock music")
            st.write("- Rap / Hip-hop")
            st.write("- Heavy beats")

        elif emotion == "Relaxed":
            st.markdown("🌿 **Chill and breathe**")
            st.write("- Ambient music")
            st.write("- Instrumentals")
            st.write("- Chill lo-fi")

        else:
            st.markdown("🎵 **Mixed mood detected — try these!**")
            st.write("- Indie music")
            st.write("- Random playlists")
            st.write("- Discover new artists")
