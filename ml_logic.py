# ml_logic.py - Updated Version for Day 3
import pandas as pd
import google.generativeai as genai
import random
import os
import sys
import time # NEW: For latency profiling
from dotenv import load_dotenv

# --- 1. SETUP: CONFIGURE API AND LOAD MODELS/DATA ---

# SECURELY load the API key from a .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model = None

if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set in your .env file.")
    sys.exit()
else:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Gemini API configured successfully!")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")

# Load the local datasets
try:
    moods_df = pd.read_csv('moods.csv')
    affirmations_df = pd.read_csv('affirmations.csv', engine='python')
    print("Local data files loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    sys.exit()

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
    "Urgent": "https://open.spotify.com/playlist/37i9dQZF1DWZrc3MTCkPzn",
    "Neutral": "https://open.spotify.com/playlist/37i9dQZF1DX8Uebhn9wzrS"
}

moods_df['mood_bucket'] = moods_df['mood_label'].map(mood_bucket_map).fillna('Neutral')
full_df = pd.merge(moods_df, affirmations_df, left_on='mood_label', right_on='mood_tag')
print("Data successfully merged and buckets applied!")


# --- 3. CORE AI AND LOGIC FUNCTIONS ---

# NEW (Day 3): Safety check function
def detect_risk(user_text: str) -> str:
    """
    Uses a keyword list to perform a simple risk assessment.
    Returns 'high' if a risk keyword is found, otherwise 'low'.
    """
    risk_keywords = ["suicide", "kill myself", "hurt myself", "end my life", "hopeless"]
    for keyword in risk_keywords:
        if keyword in user_text.lower():
            return "high"
    return "low"

def classify_mood_with_gemini(user_text: str) -> str:
    if not model:
        print("Warning: Gemini model not available. Falling back to 'Neutral'.")
        return "Neutral"
    if not user_text.strip():
        return "Neutral"

    mood_buckets = list(mood_to_playlist_map.keys())
    prompt = f"""Analyze the sentiment of the following user text. Classify it into ONE of the following categories: {', '.join(mood_buckets)}. Return only the single category name.
    User Text: "{user_text}"
    Category:"""

    try:
        # MODIFIED (Day 3): Added latency profiling
        start_time = time.time()
        response = model.generate_content(prompt)
        end_time = time.time()
        print(f"DEBUG: Gemini API call took {end_time - start_time:.4f} seconds.")

        detected_bucket = response.text.strip()
        if detected_bucket in mood_buckets:
            return detected_bucket
        else:
            for bucket in mood_buckets:
                if bucket in detected_bucket:
                    return bucket
            return "Neutral"
    except Exception as e:
        print(f"An error with Gemini API: {e}")
        return "Neutral"

# MODIFIED (Day 3): Updated to include risk detection
def get_affirmation_for_text(user_text: str) -> dict:
    # First, perform risk detection.
    risk_level = detect_risk(user_text)

    # If risk is high, return a special helpline response immediately.
    if risk_level == 'high':
        return {
            "mood_bucket": "Urgent",
            "risk": "high",
            "action": "show_helpline",
            "affirmation": "It sounds like you're going through a lot. Please know that help is available and you are not alone.",
            "helpline": {"name": "National Suicide Prevention Lifeline", "number": "988"},
            "safety_flag": "flag"
        }
    
    # If risk is low, proceed with normal mood classification.
    mood_bucket = classify_mood_with_gemini(user_text)
    possible_affirmations = full_df[full_df['mood_bucket'] == mood_bucket].to_dict('records')

    if possible_affirmations:
        selected_affirmation = random.choice(possible_affirmations)
        affirmation_text = selected_affirmation['text']
        safety_flag = selected_affirmation['safety_flag']
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

# NEW (Day 3): Trend visualization endpoint logic
def get_mood_trends(userId: str, period: str) -> dict:
    # This function uses FAKE data. Member 2 (Backend) will help connect this to the real database.
    fake_check_ins = [
        {"date": "2025-09-12", "mood_bucket": "Anxious"}, {"date": "2025-09-13", "mood_bucket": "Sad"},
        {"date": "2025-09-14", "mood_bucket": "Happy"}, {"date": "2025-09-14", "mood_bucket": "Happy"},
        {"date": "2025-09-15", "mood_bucket": "Anxious"}, {"date": "2025-09-16", "mood_bucket": "Happy"},
        {"date": "2025-09-17", "mood_bucket": "Neutral"}, {"date": "2025-09-18", "mood_bucket": "Anxious"},
    ]
    trends_df = pd.DataFrame(fake_check_ins)
    trends_df['date'] = pd.to_datetime(trends_df['date'])
    daily_counts = trends_df.groupby([trends_df['date'].dt.date, 'mood_bucket']).size().unstack(fill_value=0)
    
    labels = [date.strftime('%b %d') for date in daily_counts.index]
    datasets = []
    for mood in daily_counts.columns:
        datasets.append({"label": mood, "data": daily_counts[mood].tolist()})
        
    return {
      "userId": userId, "period": period,
      "trendData": {"labels": labels, "datasets": datasets},
      "averageSentimentScore": 0.65 # Placeholder value
    }


# --- 4. TESTING THE NEW FUNCTIONS ---

if __name__ == "__main__":
    print("\n--- Testing Day 3 Features ---")

    # Test 1: Normal, safe input
    test_text_1 = "I am feeling pretty good about my presentation today."
    print(f"\nInput: '{test_text_1}'")
    print("Affirmation Result:", get_affirmation_for_text(test_text_1))

    # Test 2: High-risk input
    test_text_2 = "I feel hopeless and want to hurt myself."
    print(f"\nInput: '{test_text_2}'")
    print("Affirmation Result:", get_affirmation_for_text(test_text_2))
    
    # Test 3: Mood trends endpoint
    print("\n--- Testing Mood Trends Endpoint ---")
    trends_result = get_mood_trends("demo_user", "7d")
    import json

    print("Trends Result (JSON):", json.dumps(trends_result, indent=2))
