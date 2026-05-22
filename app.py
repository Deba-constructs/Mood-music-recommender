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

# All 13 labels from the dataset
mood_music = {
    "happiness":  ("🎶 You're feeling great!",      ["Pop hits", "Party songs", "Dance music"]),
    "sadness":    ("🎧 Take it slow",               ["Soft acoustic", "Emotional songs", "Lo-fi beats"]),
    "anger":      ("🔥 Release the energy",         ["Rock", "Rap", "High BPM tracks"]),
    "enthusiasm": ("⚡ You're pumped!",             ["EDM", "Workout music", "Hype tracks"]),
    "neutral":    ("🌿 Chill mood",                ["Indie", "Background music", "Soft playlists"]),
    "worry":      ("🧘 Relax a bit",               ["Calm piano", "Meditation music", "Slow instrumentals"]),
    "love":       ("💖 Feeling romantic",           ["Love songs", "R&B", "Soft pop"]),
    "surprise":   ("😲 Something unexpected!",     ["Eclectic mix", "Discover Weekly", "New releases"]),
    "fun":        ("😄 Keep the fun going!",        ["Feel-good hits", "Comedy rap", "Upbeat indie"]),
    "hate":       ("😤 Shake it off",              ["Punk rock", "Metal", "Vent playlist"]),
    "boredom":    ("😴 Spice it up",              ["Shuffle mix", "Top charts", "Random genres"]),
    "relief":     ("😌 Breathe easy",             ["Ambient", "Acoustic chill", "Nature sounds"]),
    "empty":      ("🌑 Quiet moment",             ["Instrumental", "Post-rock", "Silence fillers"]),
}

user_input = st.text_input("💭 How are you feeling today?")

if st.button("🎯 Recommend Music"):
    if not user_input.strip():
        st.warning("⚠️ Please enter something!")
    else:
        vec = vectorizer.transform([user_input])
        emotion = model.predict(vec)[0]

        # Show confidence scores
        proba = model.predict_proba(vec)[0]
        confidence = max(proba) * 100
        top3 = sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)[:3]

        st.success(f"🧠 Detected Mood: **{emotion.capitalize()}** ({confidence:.0f}% confidence)")

        # Show top 3 mood probabilities
        with st.expander("📊 See mood breakdown"):
            for label, prob in top3:
                st.progress(float(prob), text=f"{label.capitalize()}: {prob*100:.1f}%")

        title, songs = mood_music.get(
            emotion,
            ("🎵 Try something new", ["Random playlist", "Discover music"])
        )

        st.markdown(f"**{title}**")
        for song in songs:
            st.write(f"- {song}")