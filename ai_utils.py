import google.generativeai as genai
from googletrans import Translator

# Initialize Gemini
genai.configure(api_key="AIzaSyACwPlnTCOgxDQQh80g_DNld7cj590bjW0")

model = genai.GenerativeModel("gemini-2.0-flash-lite")

def get_solution(question,   language="English"):
    
    translator = Translator()

    prompt = f"""
You are an expert tutor for competitive exam preparation ( Math, Physics, Chemistry, Biology, History, Computer Science, English, GeoGeography subjects). 
Provide a clear, step-by-step solution for the problems provided and give information for descriptive questions. 
Use LaTeX formatting for equations where necessary.

Problem: {question}

Solution:
"""

    try:
        response = model.generate_content(prompt)

        # Print full response to debug
        print(response)

        # Safely extract text depending on SDK version
        if hasattr(response, 'text') and response.text:
            solution = response.text
        elif hasattr(response, 'candidates'):
            solution = response.candidates[0].content.parts[0].text
        else:
            solution = "No response generated."

    except Exception as e:
        solution = f"Error generating solution: {e}"

    # Translate if needed
    if language != "English":
        try:
            translation = translator.translate(solution, dest=language.lower()[:2])
            solution = translation.text
        except Exception as e:
            solution += f"\n\n(Translation error: {e})"

    return solution