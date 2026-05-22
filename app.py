import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Mood Music Recommender", page_icon="🎧")

st.title("🎧 Mood-Based Music Recommender")
st.write("Tell me how you're feeling and I'll suggest the vibe 🎶")

# Input
user_input = st.text_input("How are you feeling?")

# Button
if st.button("Recommend"):
    if user_input.strip() == "":
        st.warning("⚠️ Bro at least type something 😭")
    else:
        text = user_input.lower()

        # 🔥 Manual fixes for weak model cases
        if "bored" in text:
            emotion = "Bored"
        else:
            vec = vectorizer.transform([user_input])
            emotion = model.predict(vec)[0]

        st.subheader(f"🧠 Detected Mood: {emotion}")

        # 🎵 Better recommendation system
        if emotion.lower() == "happy":
            st.success("🎶 Play upbeat songs!")
            st.write("Try: Pop, Dance, EDM")
        
        elif emotion.lower() == "sad":
            st.info("🎧 Try calm or emotional songs")
            st.write("Try: Lo-fi, Acoustic, Sad songs")
        
        elif emotion.lower() == "angry":
            st.error("🔥 Go for intense music")
            st.write("Try: Rock, Rap, Metal")
        
        elif emotion.lower() == "bored":
            st.write("😴 You need vibes bro")
            st.write("Try: Chill, Indie, Lo-fi")
        
        else:
            st.write("🌿 Relax with chill vibes")
            st.write("Try: Ambient, Instrumental")