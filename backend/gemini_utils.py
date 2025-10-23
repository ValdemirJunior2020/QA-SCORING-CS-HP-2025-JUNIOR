import google.generativeai as genai

# ✅ Your real Gemini API key
GEMINI_API_KEY = "AIzaSyDW9mmnncI33SqwVQ7QVXWwjvrwggGU30Y"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def analyze_with_gemini(transcript: str):
    """
    Analyze the call transcript using Gemini AI and return feedback.
    """

    prompt = f"""
You are a professional Quality Assurance Coach for HotelPlanner's call centers.

Analyze the following call transcript and return:
1. The **reason for the call**
2. What the **agent did wrong**, if anything
3. How the **agent can improve**
4. The agent's **tone and professionalism**

Format the response as clear bullet points.
Keep it short and structured.

CALL TRANSCRIPT:
{transcript}
    """

    try:
        # ⚙️ Use your available model
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        response = model.generate_content(prompt)
        return response.text or "No response generated."
    except Exception as e:
        return f"Error analyzing with Gemini: {str(e)}"
