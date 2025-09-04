# ml_logic.py - Corrected and Secure Version
import pandas as pd
import google.generativeai as genai
import random
import os   # Added for handling environment variables
import sys  # Added to allow the script to exit gracefully

# --- 1. SETUP: CONFIGURE API AND LOAD MODELS/DATA ---

# SECURELY load the API key from an environment variable
api_key = os.getenv("AIzaSyAWSc-vT9iKzE36cOQ4F6sjujKxlg77c9U")
model = None # Initialize model as None

if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the environment variable before running the script.")
    sys.exit() # Exit if the API key is not found
else:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        print("Gemini API configured successfully!")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        # model remains None, functions will handle this case

# Load the local datasets
try:
    moods_df = pd.read_csv('moods.csv')
    affirmations_df = pd.read_csv('affirmations.csv', engine='python')
    print("Local data files loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    print("Make sure 'moods.csv' and 'affirmations.csv' are in the same directory as the script.")
    sys.exit() # STOPS the script if files are not found, preventing the NameError

# --- 2. DATA PROCESSING AND MAPPING ---

mood_bucket_map = {
    "happy": "Happy", "excited": "Happy", "cheerful": "Happy", "joyful": "Happy", "perky": "Happy", "sunny": "Happy", "content": "Happy",
    "sad": "Sad", "depressed": "Sad", "lonely": "Sad", "gloomy": "Sad", "heartbroken": "Sad", "melancholy": "Sad", "dejected": "Sad",
    "anxious": "Anxious", "stressed": "Anxious", "overwhelmed": "Anxious", "tense": "Anxious", "nervous": "Anxious", "confused": "Anxious",
    "angry": "Angry", "hostile": "Angry", "resentful": "Angry", "betrayed": "Angry", "jealous": "Angry", "agitated": "Angry",
    "fearful": "Fearful", "scared": "Fearful", "terrified": "Fearful", "apprehensive": "Fearful", "intimidated": "Fearful",
    "suicidal": "Urgent", "horrified": "Urgent", "panicked": "Urgent"
}

mood_to_playlist_map = {
    "Happy": "https://open.spotify.com/playlist/37i9dQZF1DX1g0fEwBUAmQ",
    "Sad": "https://open.spotify.com/playlist/37i9dQZF1DWSqBruwoIXjL",
    "Anxious": "https://open.spotify.com/playlist/37i9dQZF1DWUvQoIOFMFUT",
    "Angry": "https://open.spotify.com/playlist/37i9dQZF1D Sextonvr0Mihs",
    "Fearful": "https://open.spotify.com/playlist/37i9dQZF1DWZrc3MTCkPzn",
    "Urgent": "https://open.spotify.com/playlist/37i9dQZF1DWZrc3MTCkPzn", # A safe, calming choice
    "Neutral": "https://open.spotify.com/playlist/37i9dQZF1DX8Uebhn9wzrS"
}

moods_df['mood_bucket'] = moods_df['mood_label'].map(mood_bucket_map).fillna('Neutral')
full_df = pd.merge(moods_df, affirmations_df, left_on='mood_label', right_on='mood_tag')
print("Data successfully merged and buckets applied!")


# --- 3. CORE AI AND LOGIC FUNCTIONS ---

def classify_mood_with_gemini(user_text: str) -> str:
    if not model: # Checks if the API model failed to load
        print("Warning: Gemini model not available. Falling back to 'Neutral'.")
        return "Neutral"
    if not user_text.strip():
        return "Neutral"

    mood_buckets = list(mood_to_playlist_map.keys())
    prompt = f"""Analyze the sentiment of the following user text. Classify it into ONE of the following categories: {', '.join(mood_buckets)}. Return only the single category name.
    User Text: "{user_text}"
    Category:"""

    try:
        response = model.generate_content(prompt)
        detected_bucket = response.text.strip()
        if detected_bucket in mood_buckets:
            return detected_bucket
        else:
            # Fallback check in case the model adds extra text
            for bucket in mood_buckets:
                if bucket in detected_bucket:
                    return bucket
            return "Neutral"
    except Exception as e:
        print(f"An error with Gemini API: {e}")
        return "Neutral"

def get_affirmation_for_text(user_text: str) -> dict:
    mood_bucket = classify_mood_with_gemini(user_text)
    possible_affirmations = full_df[full_df['mood_bucket'] == mood_bucket].to_dict('records')

    if possible_affirmations:
        selected_affirmation = random.choice(possible_affirmations)
        affirmation_text = selected_affirmation['text']
        safety_flag = selected_affirmation['safety_flag']
        if safety_flag == 'flag' or mood_bucket == 'Urgent':
            affirmation_text = "It sounds like you're going through a lot. Please consider reaching out to a support line. You are not alone."
            safety_flag = 'flag'
        return {
            "mood_bucket": mood_bucket,
            "affirmation": affirmation_text.strip(),
            "safety_flag": safety_flag
        }
    else:
        return {
            "mood_bucket": mood_bucket,
            "affirmation": "Thank you for sharing. Remember to be kind to yourself today.",
            "safety_flag": "safe"
        }

def get_music_recommendation(user_text: str) -> dict:
    mood_bucket = classify_mood_with_gemini(user_text)
    playlist_url = mood_to_playlist_map.get(mood_bucket, mood_to_playlist_map["Neutral"])
    return {
        "mood": mood_bucket,
        "playlist_url": playlist_url
    }


# --- 4. TESTING THE NEW AI-POWERED FUNCTIONS ---

if __name__ == "__main__":
    print("\n--- Testing with more complex sentences ---")

    test_text_1 = "I have a huge project due tomorrow and I haven't even started it yet."
    print(f"\nInput: '{test_text_1}'")
    print("Affirmation Result:", get_affirmation_for_text(test_text_1))
    print("Music Result:", get_music_recommendation(test_text_1))

    test_text_2 = "I just got the best news ever, I can't stop smiling!"
    print(f"\nInput: '{test_text_2}'")
    print("Affirmation Result:", get_affirmation_for_text(test_text_2))
    print("Music Result:", get_music_recommendation(test_text_2))