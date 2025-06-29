import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import json
import openai
import google.generativeai as genai
import pdfplumber
from PIL import Image
import pytesseract
import io
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS (restrict origins in production)

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    USE_OPENAI_FOR_ESSAY_SUMMARIZE = True
    USE_GEMINI_FOR_QUIZ = True
    OPENAI_MODEL = "gpt-3.5-turbo"
    GEMINI_MODEL = "gemini-pro"
    MAX_TOKENS = 500
    ALLOWED_FILE_TYPES = ['pdf', 'jpg', 'jpeg', 'png', 'txt']

# Initialize APIs
if Config.OPENAI_API_KEY:
    openai.api_key = Config.OPENAI_API_KEY
else:
    logger.error("OPENAI_API_KEY not found in .env. OpenAI API will not be available.")

if Config.GOOGLE_API_KEY:
    genai.configure(api_key=Config.GOOGLE_API_KEY)
else:
    logger.error("GOOGLE_API_KEY not found in .env. Google Gemini API will not be available.")

# In-memory store for progress tracking
progress_store = {
    'quiz_scores': defaultdict(list),  # {user_id: [(timestamp, score, difficulty)]}
    'essay_feedback': defaultdict(list)  # {user_id: [(timestamp, feedback)]}
}

# Helper Functions
def extract_text_from_file(file):
    """Extract text from uploaded files (PDF, image, or text)."""
    try:
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in Config.ALLOWED_FILE_TYPES:
            return None, "Unsupported file type."

        text_content = ""
        if file_extension == 'pdf':
            with pdfplumber.open(file.stream) as pdf:
                for page in pdf.pages:
                    text_content += page.extract_text() or ""
        elif file_extension in ['jpg', 'jpeg', 'png']:
            image = Image.open(file.stream)
            text_content = pytesseract.image_to_string(image)
        elif file_extension == 'txt':
            text_content = file.stream.read().decode('utf-8')

        if not text_content.strip():
            return None, "Could not extract any meaningful text from the file."
        return text_content, None
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        return None, f"Error processing file: {e}"

def create_error_response(message, status_code):
    """Create a standardized error response."""
    return jsonify({"error": message}), status_code

# Routes
@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')

@app.route('/upload-and-summarize', methods=['POST'])
def upload_and_summarize():
    """Handle file upload and summarization."""
    if not Config.USE_OPENAI_FOR_ESSAY_SUMMARIZE or not Config.OPENAI_API_KEY:
        return create_error_response("OpenAI API not configured.", 503)

    if 'file' not in request.files or not request.files['file'].filename:
        return create_error_response("No file provided.", 400)

    file = request.files['file']
    text_content, error = extract_text_from_file(file)
    if error:
        return create_error_response(error, 400)

    try:
        response = openai.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
                {"role": "user", "content": f"Please summarize the following text: \n\n{text_content}"}
            ],
            max_tokens=Config.MAX_TOKENS
        )
        summary = response.choices[0].message.content.strip()
        logger.info(f"Summary generated for file {file.filename}")
        return jsonify({"summary": summary})
    except openai.RateLimitError as e:
        logger.error(f"OpenAI RateLimitError: {e}")
        return create_error_response("API quota exceeded. Please check your OpenAI plan and billing details at https://platform.openai.com/account/billing/overview.", 429)
    except openai.APIError as e:
        logger.error(f"OpenAI API Error: {e}")
        return create_error_response(f"OpenAI API Error: {e.args[0]}", 500)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return create_error_response(f"An unexpected error occurred: {e}", 500)

@app.route('/summarize-text', methods=['POST'])
def summarize_text():
    """Summarize provided text."""
    if not Config.USE_OPENAI_FOR_ESSAY_SUMMARIZE or not Config.OPENAI_API_KEY:
        return create_error_response("OpenAI API not configured.", 503)

    data = request.get_json()
    text_content = data.get('text', '')
    if not text_content:
        return create_error_response("No text provided.", 400)

    try:
        response = openai.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
                {"role": "user", "content": f"Please summarize the following text: \n\n{text_content}"}
            ],
            max_tokens=Config.MAX_TOKENS
        )
        summary = response.choices[0].message.content.strip()
        logger.info("Text summarization completed")
        return jsonify({"summary": summary})
    except openai.RateLimitError as e:
        logger.error(f"OpenAI RateLimitError: {e}")
        return create_error_response("API quota exceeded. Please check your OpenAI plan and billing details at https://platform.openai.com/account/billing/overview.", 429)
    except openai.APIError as e:
        logger.error(f"OpenAI API Error: {e}")
        return create_error_response(f"OpenAI API Error: {e.args[0]}", 500)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return create_error_response(f"An unexpected error occurred: {e}", 500)

@app.route('/analyze-essay', methods=['POST'])
def analyze_essay():
    """Analyze an essay for coherence, tone, grammar, and clarity."""
    if not Config.USE_OPENAI_FOR_ESSAY_SUMMARIZE or not Config.OPENAI_API_KEY:
        return create_error_response("OpenAI API not configured.", 503)

    data = request.get_json()
    essay = data.get('essay', '')
    user_id = data.get('user_id', 'anonymous')  # For progress tracking

    if not essay:
        return create_error_response("No essay provided.", 400)

    prompt = f"""Analyze the following essay for:
    1. Coherence (is it well-structured, does it flow logically?)
    2. Empathy and tone (what is the overall tone? Does it convey empathy if applicable?)
    3. Grammar and spelling (mention if there are errors, no need to list them all)
    4. Clarity and flow (is the writing clear and easy to understand? Does it transition smoothly?)

    Provide concise feedback for each point. Format the response strictly as:
    Coherence: [Feedback]
    Empathy and tone: [Feedback]
    Grammar & spelling: [Feedback]
    Clarity and flow: [Feedback]

    Essay:
    {essay}
    """

    try:
        response = openai.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI mentor providing constructive feedback on essays."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=Config.MAX_TOKENS
        )
        feedback_raw = response.choices[0].message.content.strip()

        # Parse feedback into structured data
        feedback_dict = {
            "coherence": "N/A",
            "tone": "N/A",
            "grammar": "N/A",
            "clarity": "N/A"
        }
        for line in feedback_raw.split('\n'):
            if "Coherence:" in line:
                feedback_dict["coherence"] = line.replace("Coherence:", "").strip()
            elif "Empathy and tone:" in line:
                feedback_dict["tone"] = line.replace("Empathy and tone:", "").strip()
            elif "Grammar & spelling:" in line:
                feedback_dict["grammar"] = line.replace("Grammar & spelling:", "").strip()
            elif "Clarity and flow:" in line:
                feedback_dict["clarity"] = line.replace("Clarity and flow:", "").strip()

        # Store feedback in progress store
        from datetime import datetime
        progress_store['essay_feedback'][user_id].append({
            'timestamp': datetime.now().isoformat(),
            'feedback': feedback_dict
        })

        logger.info(f"Essay analyzed for user {user_id}")
        return jsonify(feedback_dict)
    except openai.RateLimitError as e:
        logger.error(f"OpenAI RateLimitError: {e}")
        return create_error_response("API quota exceeded. Please check your OpenAI plan and billing details at https://platform.openai.com/account/billing/overview.", 429)
    except openai.APIError as e:
        logger.error(f"OpenAI API Error: {e}")
        return create_error_response(f"OpenAI API Error: {e.args[0]}", 500)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return create_error_response(f"An unexpected error occurred: {e}", 500)

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    """Generate a quiz based on provided text and difficulty."""
    if not Config.USE_GEMINI_FOR_QUIZ or not Config.GOOGLE_API_KEY:
        return create_error_response("Google Gemini API not configured.", 503)

    data = request.get_json()
    text_content = data.get('text', '')
    difficulty = data.get('difficulty', 'easy').lower()
    user_id = data.get('user_id', 'anonymous')  # For progress tracking

    if not text_content:
        return create_error_response("No text provided.", 400)

    prompt = f"""Generate a {difficulty} difficulty quiz with 3-5 questions (mix of Multiple Choice and True/False) based on the following text.
    For Multiple Choice Questions, provide 4 options (A, B, C, D) and clearly indicate the correct answer.
    For True/False questions, clearly indicate True or False as the answer.

    Text:
    {text_content}

    Format the output strictly as a JSON array of objects, where each object has "question", "options" (optional, for MCQs), and "answer" fields.
    Example for MCQ: {{"question": "What is the capital of France?", "options": ["London", "Berlin", "Paris", "Rome"], "answer": "Paris"}}
    Example for True/False: {{"question": "The sky is blue.", "answer": "True"}}
    """

    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        response = model.generate_content(prompt)
        quiz_raw = response.text.strip()

        # Parse JSON output from Gemini
        if quiz_raw.startswith('```json'):
            quiz_raw = quiz_raw[len('```json'):].strip()
        if quiz_raw.endswith('```'):
            quiz_raw = quiz_raw[:-len('```')].strip()

        quiz_data = json.loads(quiz_raw)
        logger.info(f"Quiz generated for user {user_id}, difficulty: {difficulty}")
        return jsonify({"quiz": quiz_data})
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {e}, Raw output: {quiz_raw}")
        return create_error_response("Failed to parse quiz data from AI.", 500)
    except Exception as e:
        logger.error(f"Google Gemini API Error: {e}")
        return create_error_response(f"Google Gemini API Error: {e}", 500)

@app.route('/submit-quiz', methods=['POST'])
def submit_quiz():
    """Submit quiz answers and store scores."""
    data = request.get_json()
    user_id = data.get('user_id', 'anonymous')
    answers = data.get('answers', [])  # List of {question, user_answer, correct_answer}
    difficulty = data.get('difficulty', 'easy')

    if not answers:
        return create_error_response("No answers provided.", 400)

    score = sum(1 for ans in answers if ans.get('user_answer') == ans.get('correct_answer'))
    total = len(answers)
    from datetime import datetime
    progress_store['quiz_scores'][user_id].append({
        'timestamp': datetime.now().isoformat(),
        'score': score,
        'total': total,
        'difficulty': difficulty
    })

    logger.info(f"Quiz submitted for user {user_id}, score: {score}/{total}")
    return jsonify({"score": score, "total": total})

@app.route('/progress', methods=['GET'])
def get_progress():
    """Retrieve user progress (quiz scores and essay feedback)."""
    user_id = request.args.get('user_id', 'anonymous')
    return jsonify({
        'quiz_scores': progress_store['quiz_scores'][user_id],
        'essay_feedback': progress_store['essay_feedback'][user_id]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)