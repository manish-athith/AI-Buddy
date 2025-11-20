****AI Study Buddy with Real ML ****

An interactive Streamlit-based AI study assistant that combines traditional machine learning classifiers with LLM-powered answers to give students age-appropriate, subject-aware, and intent-specific explanations. 

The app analyzes each question using real ML models, decides who the user is (age group), what they’re asking about (subject), and why they’re asking (intent), then crafts an optimized prompt for a Groq-hosted LLM (LLaMA/Mixtral) to generate a high-quality educational response.


** Key Features:**

ML-powered question classification

Predicts:

  age_group: elementary / middle_school / high_school

  subject: science / math / history / language

  intent: homework_help / concept_explanation / factual_doubt / creative_story

  language (currently English)
        

**Smart prompt engineering:**

  Uses classification output to select a prompt template tailored to:

  learner level (kid, teen, high school)

  subject domain (science, math, history, language)

  intent (concept explanation, homework help, creative story, etc.)

**Real LLM integration via Groq API:**

  Supports multiple models:

  llama-3.1-8b-instant

  llama-3.1-70b-versatile

  mixtral-8x7b-32768


**Rule-based fallback:**

If ML models are unavailable, falls back gracefully to keyword-based rules.


**Clean, interactive UI (Streamlit):**

  Sidebar configuration for:

  Classification method (ML vs Rule-Based)

  Groq API key

  Model selection

  Simple analytics (total questions, success rate)


**Feedback & Analytics:**


  Thumbs up / thumbs down feedback on responses

  Tracks success rate based on feedback

  Shows recent conversation history with metadata



**How It Works:**

1)User enters a question in the text area

2)Depending on the sidebar selection:

ML mode:

Uses a pre-trained TF-IDF vectorizer and classifiers (age_classifier, subject_classifier, intent_classifier, language_classifier) loaded via joblib from the     models/ folder.

Rule-based mode:

Uses simple keyword rules to guess age group, subject, intent, and language.

3)The app builds a prompt using the PROMPT_TEMPLATES dictionary, e.g.:

“You are a middle school science teacher…”

“You are a math tutor for high school students…”

4)The optimized prompt is sent to the Groq Chat Completions API, which returns the final AI-generated explanation.

5)The app:

Displays classification results and (if in ML mode) confidence scores

Renders the AI response in a styled box

Stores the interaction and optional feedback for analytics


**Tech Stack:**

Frontend / UI: Streamlit

ML Runtime:

 joblib (for loading pre-trained models)

 pandas, numpy (for data handling)

LLM Backend: Groq API (OpenAI-compatible chat completions)

HTTP: requests

Language: Python 3

**Project Structure:**

.
├── ai_buddy_ml.py          # Main Streamlit app
├── main.ipynb              # Notebook (training / experimentation, etc.)
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── age_classifier.pkl
│   ├── subject_classifier.pkl
│   ├── intent_classifier.pkl
│   └── language_classifier.pkl
├── requirements.txt        # (recommended) Python dependencies
└── README.md               # Description


**Setup & Installation:**

1. Clone the repository:
   git clone https://github.com/<manish-athith>/<AI-Buddy>.git
   cd <AI-Buddy>
   
2. Create and activate a virtual environment (optional but recommended)
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate

3. Install dependencies:
Create a requirements.txt (if you don’t already have one) with at least:

  streamlit
  pandas
  numpy
  joblib
  requests
  scikit-learn

Then install:
pip install -r requirements.txt

4. Add ML models

Make sure you have the trained models saved in the models/ directory:

models/tfidf_vectorizer.pkl

models/age_classifier.pkl

models/subject_classifier.pkl

models/intent_classifier.pkl

models/language_classifier.pkl

If you trained these in main.ipynb, export them with joblib.dump(...) into the models/ folder.

5. Get your Groq API key

Sign up / log in to Groq Cloud

Create an API key (starts with gsk_...)

You don’t need to export it as an environment variable—
the key is entered directly in the Streamlit sidebar.

**Running the App:**

streamlit run ai_buddy_ml.py


**Author**

Manish Choudhary

GitHub: https://github.com/<manish-athith>

LinkedIn: https://www.linkedin.com/in/manish-choudhary110904/
Email: manish.athith@gmail.com





