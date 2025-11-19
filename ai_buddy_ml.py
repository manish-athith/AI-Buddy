# ai_buddy_ml.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import requests
import random

# Page configuration
st.set_page_config(
    page_title="AI Study Buddy with Real ML",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Smart prompt templates for LLM
PROMPT_TEMPLATES = {
    ('science', 'factual_doubt', 'elementary'): 
        "You are a friendly science teacher explaining to an 8-year-old child. Explain this in simple, fun terms with analogies and examples they can relate to. Keep it engaging and educational: {question}",
    
    ('science', 'factual_doubt', 'middle_school'): 
        "You are a middle school science teacher. Provide a clear, factual explanation that's engaging for teenagers. Include interesting facts and real-world applications: {question}",
    
    ('science', 'factual_doubt', 'high_school'): 
        "You are a high school science teacher. Give a detailed, scientific explanation with key concepts and principles. Make it comprehensive but accessible: {question}",
    
    ('science', 'concept_explanation', 'elementary'): 
        "You are explaining a complex science concept to a young child. Use simple analogies, stories, and examples from their daily life. Make it magical and fun: {question}",
    
    ('science', 'concept_explanation', 'middle_school'): 
        "You are making science exciting for teenagers. Explain this concept with cool examples, real-world applications, and why it matters: {question}",
    
    ('math', 'homework_help', 'middle_school'): 
        "You are a patient math tutor helping a middle school student. Provide step-by-step guidance and explain each step clearly. Show the reasoning behind each step: {question}",
    
    ('math', 'homework_help', 'high_school'): 
        "You are a math tutor for high school students. Solve this problem with detailed working, explain the underlying concepts, and show alternative approaches: {question}",
    
    ('language', 'creative_story', 'elementar'): 
        "You are a children's story writer. Create a short, imaginative, age-appropriate story that's engaging and has a positive message: {question}",
    
    ('language', 'creative_story', 'middle_school'): 
        "You are a creative writing teacher. Write an engaging story that would interest teenagers, with relatable characters and an interesting plot: {question}",
    
    ('history', 'concept_explanation', 'middle_school'): 
        "You are a history teacher making the past come alive for teenagers. Explain this in an engaging, story-like way with interesting facts: {question}",
    
    ('history', 'concept_explanation', 'high_school'): 
        "You are a history professor. Provide a detailed analysis with historical context, significance, and multiple perspectives: {question}"
}

# Load ML models
@st.cache_resource
def load_ml_models():
    """Load pre-trained ML models"""
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        age_classifier = joblib.load('models/age_classifier.pkl')
        subject_classifier = joblib.load('models/subject_classifier.pkl')
        intent_classifier = joblib.load('models/intent_classifier.pkl')
        language_classifier = joblib.load('models/language_classifier.pkl')
        
        return {
            'vectorizer': vectorizer,
            'age_group': age_classifier,
            'subject': subject_classifier,
            'intent': intent_classifier,
            'language': language_classifier
        }
    except Exception as e:
        st.error(f"❌ Error loading ML models: {str(e)}")
        return None

# ML-based classification function
def classify_question_ml(question, models):
    """Classify using actual trained ML models"""
    # Convert question to features
    question_vectorized = models['vectorizer'].transform([question])
    
    # Get predictions from all classifiers
    age_pred = models['age_group'].predict(question_vectorized)[0]
    subject_pred = models['subject'].predict(question_vectorized)[0]
    intent_pred = models['intent'].predict(question_vectorized)[0]
    language_pred = models['language'].predict(question_vectorized)[0]
    
    return age_pred, subject_pred, intent_pred, language_pred

# Rule-based classification fallback
def classify_question_rule_based(question):
    """Rule-based classification fallback"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['child', 'kid', 'young', '8 year', '10 year']):
        age_group = 'elementary'
    elif any(word in question_lower for word in ['teen', 'high school', 'advanced']):
        age_group = 'high_school'
    else:
        age_group = 'middle_school'
    
    if any(word in question_lower for word in ['math', 'solve', 'equation', 'calculate']):
        subject = 'math'
    elif any(word in question_lower for word in ['history', 'war', 'revolution', 'ancient']):
        subject = 'history'
    elif any(word in question_lower for word in ['story', 'write', 'poem', 'creative']):
        subject = 'language'
    else:
        subject = 'science'
    
    if any(word in question_lower for word in ['story', 'write', 'create', 'tale']):
        intent = 'creative_story'
    elif any(word in question_lower for word in ['solve', 'calculate', 'help with', 'how to']):
        intent = 'homework_help'
    elif any(word in question_lower for word in ['explain', 'how does', 'concept', 'why does']):
        intent = 'concept_explanation'
    else:
        intent = 'factual_doubt'
    
    language = 'english'
    
    return age_group, subject, intent, language

# Get AI response from Groq
def get_ai_response(prompt, model_choice="llama-3.1-8b-instant", api_key=""):
    """Get real AI response from Groq API"""
    try:
        if not api_key:
            return "❌ Please set your Groq API key to get real AI answers!"
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful, engaging, and educational assistant for students."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "model": model_choice,
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"🚫 Error: {str(e)}"

def main():
    st.title("🤖 AI Study Buddy with Real ML")
    st.markdown("### Now with Machine Learning Classification!")
    
    # Custom CSS
    st.markdown("""
    <style>
    .ai-response-box {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        font-size: 16px;
        line-height: 1.6;
        margin: 10px 0;
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load ML models
    ml_models = load_ml_models()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Classification method
        if ml_models:
            classification_method = st.radio(
                "Classification Method:",
                ["🤖 ML Models", "📝 Rule-Based"],
                help="ML Models use trained classifiers, Rule-Based uses keyword matching"
            )
        else:
            classification_method = "📝 Rule-Based"
            st.warning("Using Rule-Based classification (ML models not available)")
        
        # API Key - Fixed session state handling
        st.header("🔑 API Setup")
        api_key_input = st.text_input(
            "Groq API Key:", 
            value=st.session_state.api_key,
            type="password", 
            placeholder="gsk_...", 
            key="api_key_input"
        )
        
        # Update session state only when input changes
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            
        if st.session_state.api_key:
            st.success("✅ API Key set!")
        
        # Model selection
        st.header("🤖 AI Model")
        model_choice = st.selectbox(
            "Choose AI Model:",
            ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
            index=0
        )
        
        # Analytics
        st.header("📊 Analytics")
        st.metric("Total Questions", len(st.session_state.conversation_history))
        
        if st.session_state.feedback_data:
            positive = len([f for f in st.session_state.feedback_data if f['rating'] == 'positive'])
            rate = (positive / len(st.session_state.feedback_data)) * 100
            st.metric("Success Rate", f"{rate:.1f}%")
    
    # Main interface
    st.markdown("---")
    
    # Question input
    question = st.text_area(
        "**Ask your question:**",
        placeholder="Type your question here...\nExamples:\n• Why is the sky blue?\n• How to solve 2x + 5 = 15?\n• Write a story about friendship",
        height=120
    )
    
    if st.button("🚀 Analyze & Get AI Answer") and question:
        if not st.session_state.api_key:
            st.error("❌ Please enter your Groq API Key in the sidebar!")
            return
            
        with st.spinner("🔍 Analyzing with ML and generating AI response..."):
            # Step 1: Classify the question
            if classification_method == "🤖 ML Models" and ml_models:
                age, subject, intent, language = classify_question_ml(question, ml_models)
                method_used = "🤖 ML Classification"
                
                # Show confidence scores
                question_vectorized = ml_models['vectorizer'].transform([question])
                age_probs = ml_models['age_group'].predict_proba(question_vectorized)[0]
                subject_probs = ml_models['subject'].predict_proba(question_vectorized)[0]
                
            else:
                age, subject, intent, language = classify_question_rule_based(question)
                method_used = "📝 Rule-Based Classification"
            
            # Step 2: Create optimized prompt
            template_key = (subject, intent, age)
            if template_key in PROMPT_TEMPLATES:
                optimized_prompt = PROMPT_TEMPLATES[template_key].format(question=question)
            else:
                optimized_prompt = f"Provide a helpful educational response to: {question}"
            
            # Step 3: Get AI response
            ai_response = get_ai_response(optimized_prompt, model_choice, st.session_state.api_key)
            
            # Store conversation
            conversation_entry = {
                'question': question,
                'classification_method': method_used,
                'classification': {'age_group': age, 'subject': subject, 'intent': intent, 'language': language},
                'optimized_prompt': optimized_prompt,
                'ai_model': model_choice,
                'ai_response': ai_response,
                'feedback': None
            }
            st.session_state.conversation_history.append(conversation_entry)
        
        # Display results
        st.markdown("---")
        
        # Show classification results
        st.subheader("🎯 Classification Results")
        st.success(f"Method: {method_used}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Age Level", age.title())
        col2.metric("Subject", subject.title())
        col3.metric("Intent", intent.replace('_', ' ').title())
        col4.metric("AI Model", model_choice)
        
        # Show confidence scores if using ML
        if classification_method == "🤖 ML Models" and ml_models:
            with st.expander("🔬 ML Confidence Scores"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Age Group Confidence:**")
                    for class_name, prob in zip(ml_models['age_group'].classes_, age_probs):
                        st.write(f"{class_name}: {prob:.2%}")
                with col2:
                    st.write("**Subject Confidence:**")
                    for class_name, prob in zip(ml_models['subject'].classes_, subject_probs):
                        st.write(f"{class_name}: {prob:.2%}")
        
        # Show optimized prompt
        with st.expander("🔍 See Optimized Prompt"):
            st.code(optimized_prompt)
        
        # Show AI response
        st.subheader("💡 AI Response")
        if any(error in ai_response for error in ['❌', '🚫', 'Error:']):
            st.error(ai_response)
        else:
            st.markdown(f'<div class="ai-response-box">{ai_response}</div>', unsafe_allow_html=True)
            st.success("✅ Real AI Response Generated!")
        
        # Feedback system
        st.markdown("---")
        st.subheader("📊 Feedback")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful Response", key="feedback_positive"):
                st.session_state.conversation_history[-1]['feedback'] = 'positive'
                st.session_state.feedback_data.append({'rating': 'positive'})
                st.success("✅ Thanks for your feedback!")
        with col2:
            if st.button("👎 Needs Improvement", key="feedback_negative"):
                st.session_state.conversation_history[-1]['feedback'] = 'negative'
                st.session_state.feedback_data.append({'rating': 'negative'})
                st.info("📝 Thanks! We'll improve.")
    
    # Conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("📝 Conversation History")
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-3:])):
            with st.expander(f"Q: {conv['question'][:50]}...", expanded=i==0):
                st.write(f"**Classification Method:** {conv['classification_method']}")
                st.write(f"**Analysis:** {conv['classification']['age_group'].title()} student, {conv['classification']['subject'].title()}, {conv['classification']['intent'].replace('_', ' ').title()}")
                st.write(f"**AI Model:** {conv['ai_model']}")
                
                if any(error in conv['ai_response'] for error in ['❌', '🚫', 'Error:']):
                    st.error(f"**Response:** {conv['ai_response']}")
                else:
                    st.write(f"**Response:** {conv['ai_response']}")
                
                if conv['feedback']:
                    st.write(f"**Your Feedback:** {'✅ Helpful' if conv['feedback'] == 'positive' else '📝 Needs Improvement'}")

    # Demo instructions
    st.markdown("---")
    with st.expander("🎯 Demo Instructions"):
        st.markdown("""
        **Test the ML System:**
        
        1. **Enter your Groq API key** in the sidebar
        2. **Select '🤖 ML Models'** for classification
        3. **Try these test questions:**
           - "Why is the sky blue?" (Science fact)
           - "How to solve 2x + 5 = 15?" (Math homework)
           - "Write a story about friendship" (Creative writing)
        
        **Compare Methods:**
        - Switch between **ML Models** and **Rule-Based**
        - See **confidence scores** with ML
        - Notice how ML handles complex questions better
        """)

if __name__ == "__main__":
    main()