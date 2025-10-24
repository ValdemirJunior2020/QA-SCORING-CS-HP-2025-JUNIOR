import google.generativeai as genai

# ✅ Gemini API key (replace with yours if needed)
GEMINI_API_KEY = "AIzaSyDW9mmnncI33SqwVQ7QVXWwjvrwggGU30Y"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def analyze_with_gemini(transcript: str):
    """
    Analyze transcript using Gemini and return summarized QA feedback.
    """
    prompt = f"""
You are a professional Quality Assurance Coach for HotelPlanner's call centers.

Analyze the following call transcript and provide:
1. The reason for the call.
2. What the agent did wrong (if anything).
3. How the agent can improve.
4. The agent's tone and professionalism.

Keep it short, objective, and easy to read.

TRANSCRIPT:
{transcript}
    """

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text or "No feedback generated."
    except Exception as e:
        return f"Error analyzing with Gemini: {e}"
