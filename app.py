import streamlit as st
import pickle

st.set_page_config(page_title="Mood Music Recommender", page_icon="🎧")

st.title("🎧 Mood-Based Music Recommender")
st.markdown("Tell me how you're feeling, and I'll suggest the perfect vibe 🎶")

@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

mood_music = {
    "happiness": ("🎶 You're feeling great!", ["Pop hits", "Party songs", "Dance music"]),
    "sadness": ("🎧 Take it slow", ["Soft acoustic", "Emotional songs", "Lo-fi beats"]),
    "anger": ("🔥 Release the energy", ["Rock", "Rap", "High BPM tracks"]),
    "enthusiasm": ("⚡ You're pumped!", ["EDM", "Workout music", "Hype tracks"]),
    "neutral": ("🌿 Chill mood", ["Indie", "Background music", "Soft playlists"]),
    "worry": ("🧘 Relax a bit", ["Calm piano", "Meditation music", "Slow instrumentals"])
}

user_input = st.text_input("💭 How are you feeling today?")

if st.button("🎯 Recommend Music"):
    if not user_input.strip():
        st.warning("⚠️ Please enter something!")
    else:
        vec = vectorizer.transform([user_input])
        emotion = model.predict(vec)[0]

        st.success(f"🧠 Detected Mood: **{emotion}**")

        title, songs = mood_music.get(
            emotion,
            ("🎵 Try something new", ["Random playlist", "Discover music"])
        )

        st.markdown(f"**{title}**")
        for song in songs:
            st.write(f"- {song}")