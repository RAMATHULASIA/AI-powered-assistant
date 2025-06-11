       
# Install required packages (run in Google Colab cell)
# !pip install streamlit gradio transformers torch accelerate plotly pandas

import streamlit as st
import gradio as gr
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datetime import date, datetime
import pandas as pd
import os
import json
import plotly.express as px
import threading
import time

# ---------------------- MODEL LOADING -----------------------
@st.cache_resource
def load_granite_model():
    """Load IBM Granite 3.3-2B Instruct model"""
    model_name = "ibm-granite/granite-3.3-2b-instruct"
    
    print("Loading IBM Granite model...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    # Create text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_length=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    print("Model loaded successfully!")
    return text_generator, tokenizer

# ---------------------- GRANITE-POWERED FUNCTIONS -----------------------
def generate_personalized_questions(topic, grade_level, num_questions=3):
    """Generate personalized questions using Granite model"""
    prompt = f"""Generate {num_questions} educational questions for {grade_level} students about {topic}. 
    Make them engaging and appropriate for the grade level. Format as a numbered list:
    
    Topic: {topic}
    Grade: {grade_level}
    Questions:"""
    
    try:
        result = granite_generator(prompt, max_new_tokens=200, num_return_sequences=1)
        return result[0]['generated_text'].split("Questions:")[-1].strip()
    except Exception as e:
        return f"Error generating questions: {str(e)}"

def generate_ai_feedback(score, subject, student_answer=""):
    """Generate personalized feedback using Granite model"""
    performance_level = "excellent" if score >= 85 else "good" if score >= 70 else "needs improvement"
    
    prompt = f"""As an AI teaching assistant, provide constructive feedback for a student who scored {score}% in {subject}. 
    Their performance level is {performance_level}. 
    Student's answer: "{student_answer}"
    
    Provide encouraging and specific feedback to help them improve:"""
    
    try:
        result = granite_generator(prompt, max_new_tokens=150, num_return_sequences=1)
        feedback = result[0]['generated_text'].split("Provide encouraging and specific feedback to help them improve:")[-1].strip()
        return feedback
    except Exception as e:
        return f"Great effort! Keep practicing to improve your understanding of {subject}."

def ai_teaching_assistant(question, context=""):
    """AI Teaching Assistant powered by Granite"""
    prompt = f"""You are an AI teaching assistant. Answer the student's question clearly and helpfully.
    
    Context: {context}
    Student Question: {question}
    
    Answer:"""
    
    try:
        result = granite_generator(prompt, max_new_tokens=200, num_return_sequences=1)
        answer = result[0]['generated_text'].split("Answer:")[-1].strip()
        return answer
    except Exception as e:
        return f"I'm sorry, I couldn't process your question right now. Please try again."

# ---------------------- UTILITY FUNCTIONS -----------------------
def save_quiz_submission(data):
    os.makedirs("quiz_submissions", exist_ok=True)
    file_path = f"quiz_submissions/{data['Name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def rubric_score(answer):
    score = 0
    if len(answer) > 10:
        score += 2
    if any(word in answer.lower() for word in ["because", "therefore", "explains", "process", "result"]):
        score += 2
    if answer[0].isupper() and answer.endswith('.'):
        score += 1
    return min(score, 5)

# ---------------------- STREAMLIT APP -----------------------
def run_streamlit_app():
    st.set_page_config(page_title="AI Education Platform", page_icon="🎓", layout="wide")
    
    # Load model
    global granite_generator, tokenizer
    granite_generator, tokenizer = load_granite_model()
    
    # ---------------------- SIDEBAR NAVIGATION -----------------------
    st.sidebar.title("🎓 AI Education Platform")
    st.sidebar.markdown("*Powered by IBM Granite AI*")
    
    section = st.sidebar.radio("Navigate to:", [
        "🏠 Home", 
        "👤 Student Profile", 
        "📝 AI Assessments", 
        "💬 AI Feedback", 
        "📤 Submission", 
        "🧠 AI Quiz Grader", 
        "📊 Performance Dashboard", 
        "🤖 AI Teaching Assistant"
    ])

    # ---------------------- HOME SECTION -----------------------
    if section == "🏠 Home":
        st.title("🎓 AI-Powered Educational Platform")
        st.markdown("### Welcome to the future of personalized learning!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚀 **Key Features**")
            st.markdown("""
            - **AI-Generated Questions**: Personalized assessments using IBM Granite
            - **Intelligent Feedback**: AI-powered performance analysis
            - **Real-time Grading**: Automated quiz evaluation
            - **Performance Tracking**: Visual progress monitoring
            - **24/7 AI Tutor**: Always available teaching assistant
            """)
        
        with col2:
            st.markdown("#### 🛠️ **Technologies**")
            st.markdown("""
            | Component | Technology |
            |-----------|------------|
            | **AI Model** | IBM Granite 3.3-2B Instruct |
            | **Framework** | Streamlit + Gradio |
            | **Platform** | Google Colab |
            | **Libraries** | Transformers, PyTorch |
            | **Visualization** | Plotly |
            """)

    # ---------------------- STUDENT PROFILE -----------------------
    elif section == "👤 Student Profile":
        st.title("👤 Student Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("📝 Student Name:")
            age = st.number_input("🎂 Age:", min_value=5, max_value=100, value=15)
            grade = st.selectbox("🎓 Grade Level:", [
                "Elementary (K-5)", "Middle School (6-8)", 
                "High School (9-12)", "College", "Graduate"
            ])
        
        with col2:
            interests = st.text_area("🎯 Learning Interests:", 
                placeholder="e.g., Mathematics, Science, Literature, History...")
            learning_style = st.selectbox("📚 Preferred Learning Style:", [
                "Visual", "Auditory", "Kinesthetic", "Reading/Writing"
            ])
        
        if st.button("💾 Save Profile", type="primary"):
            st.session_state.student_profile = {
                "name": name, "age": age, "grade": grade, 
                "interests": interests, "learning_style": learning_style
            }
            st.success(f"✅ Profile saved for {name}!")
            st.balloons()

    # ---------------------- AI ASSESSMENTS -----------------------
    elif section == "📝 AI Assessments":
        st.title("📝 AI-Generated Assessments")
        
        profile = st.session_state.get("student_profile", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("📚 Subject/Topic:", 
                value=profile.get("interests", "Mathematics").split(",")[0])
            grade_level = st.selectbox("🎓 Grade Level:", [
                "Elementary", "Middle School", "High School", "College"
            ], index=2)
        
        with col2:
            num_questions = st.slider("❓ Number of Questions:", 1, 5, 3)
            difficulty = st.select_slider("⚡ Difficulty Level:", [
                "Beginner", "Intermediate", "Advanced"
            ])
        
        if st.button("🤖 Generate AI Assessment", type="primary"):
            with st.spinner("🧠 AI is creating personalized questions..."):
                questions = generate_personalized_questions(topic, grade_level, num_questions)
                
                st.markdown("### 📋 Your Personalized Assessment")
                st.markdown(f"**Subject:** {topic} | **Level:** {grade_level}")
                st.markdown("---")
                st.markdown(questions)

    # ---------------------- AI FEEDBACK -----------------------
    elif section == "💬 AI Feedback":
        st.title("💬 AI-Powered Feedback")
        
        col1, col2 = st.columns(2)
        
        with col1:
            subject = st.selectbox("📚 Subject:", [
                "Mathematics", "Science", "English", "History", "Art"
            ])
            score = st.slider("📊 Student Score (%):", 0, 100, 75)
        
        with col2:
            student_answer = st.text_area("📝 Student's Answer (optional):", 
                placeholder="Paste the student's response here for more personalized feedback...")
        
        if st.button("🤖 Generate AI Feedback", type="primary"):
            with st.spinner("🧠 AI is analyzing performance..."):
                feedback = generate_ai_feedback(score, subject, student_answer)
                
                # Performance indicator
                if score >= 85:
                    st.success("🎉 Excellent Performance!")
                elif score >= 70:
                    st.info("👍 Good Work!")
                else:
                    st.warning("💪 Room for Improvement!")
                
                st.markdown("### 🤖 AI Feedback")
                st.markdown(feedback)

    # ---------------------- AI TEACHING ASSISTANT -----------------------
    elif section == "🤖 AI Teaching Assistant":
        st.title("🤖 AI Teaching Assistant")
        st.markdown("*Ask me anything about your studies!*")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat interface
        user_question = st.text_input("💭 Ask your question:", 
            placeholder="e.g., Explain photosynthesis, Help with algebra, What is the water cycle?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🚀 Ask AI", type="primary")
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        if ask_button and user_question:
            with st.spinner("🤖 AI is thinking..."):
                # Add context based on student profile
                profile = st.session_state.get("student_profile", {})
                context = f"Student grade: {profile.get('grade', 'High School')}, Interests: {profile.get('interests', 'General')}"
                
                response = ai_teaching_assistant(user_question, context)
                
                # Add to chat history
                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("assistant", response))
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Chat History")
            for i, (role, message) in enumerate(reversed(st.session_state.chat_history[-10:])):
                if role == "user":
                    st.markdown(f"**👤 You:** {message}")
                else:
                    st.markdown(f"**🤖 AI Assistant:** {message}")
                st.markdown("---")

    # ---------------------- QUIZ GRADER -----------------------
    elif section == "🧠 AI Quiz Grader":
        st.title("🧠 AI-Powered Quiz Grader")
        
        with st.form("quiz_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("👤 Student Name:")
                subject = st.selectbox("📚 Subject:", ["Mathematics", "Science", "English", "History"])
            
            with col2:
                quiz_type = st.selectbox("📋 Quiz Type:", ["Mixed", "Multiple Choice", "Short Answer"])
            
            # Sample questions
            st.markdown("### 📝 Sample Questions")
            
            # MCQ
            mcq_q = st.radio("🔘 What is the capital of France?", 
                ["Berlin", "Madrid", "Paris", "Rome"])
            
            # Short answer
            short_answer = st.text_area("✍️ Explain the process of photosynthesis:", 
                placeholder="Write your answer here...")
            
            # File upload
            uploaded_file = st.file_uploader("📎 Upload additional answers (TXT):", type=["txt"])
            
            submit_quiz = st.form_submit_button("🎯 Grade Quiz with AI", type="primary")
            
            if submit_quiz and name:
                with st.spinner("🤖 AI is grading your quiz..."):
                    # Calculate scores
                    mcq_score = 5 if mcq_q == "Paris" else 0
                    short_score = rubric_score(short_answer)
                    
                    file_answer = ""
                    file_score = 0
                    if uploaded_file:
                        content = uploaded_file.read().decode("utf-8")
                        file_answer = content
                        file_score = rubric_score(content)
                    
                    total_score = mcq_score + short_score + file_score
                    percentage = (total_score / 15) * 100
                    
                    # Generate AI feedback
                    ai_feedback = generate_ai_feedback(percentage, subject, short_answer)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MCQ Score", f"{mcq_score}/5")
                    with col2:
                        st.metric("Written Score", f"{short_score}/5")
                    with col3:
                        st.metric("File Score", f"{file_score}/5")
                    
                    st.markdown("### 📊 Overall Performance")
                    st.progress(percentage/100)
                    st.markdown(f"**Total Score: {total_score}/15 ({percentage:.1f}%)**")
                    
                    st.markdown("### 🤖 AI Feedback")
                    st.info(ai_feedback)
                    
                    # Save submission
                    submission_data = {
                        "Name": name, "Subject": subject, "MCQ_Score": mcq_score,
                        "Short_Score": short_score, "File_Score": file_score,
                        "Total": total_score, "Percentage": percentage,
                        "AI_Feedback": ai_feedback, "Date": str(datetime.now())
                    }
                    save_quiz_submission(submission_data)

    # ---------------------- PERFORMANCE DASHBOARD -----------------------
    elif section == "📊 Performance Dashboard":
        st.title("📊 Student Performance Dashboard")
        
        if os.path.exists("quiz_submissions") and os.listdir("quiz_submissions"):
            # Load data
            all_files = os.listdir("quiz_submissions")
            data = []
            for file in all_files:
                try:
                    with open(f"quiz_submissions/{file}", "r") as f:
                        data.append(json.load(f))
                except:
                    continue
            
            if data:
                df = pd.DataFrame(data)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Student selector
                student_names = df['Name'].unique()
                selected_student = st.selectbox("👤 Select Student:", student_names)
                
                # Filter data
                student_data = df[df['Name'] == selected_student].sort_values("Date")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Quizzes", len(student_data))
                with col2:
                    avg_score = student_data['Percentage'].mean()
                    st.metric("Average Score", f"{avg_score:.1f}%")
                with col3:
                    best_score = student_data['Percentage'].max()
                    st.metric("Best Score", f"{best_score:.1f}%")
                with col4:
                    recent_score = student_data['Percentage'].iloc[-1]
                    st.metric("Recent Score", f"{recent_score:.1f}%")
                
                # Performance chart
                fig = px.line(student_data, x="Date", y="Percentage", 
                            title=f"📈 Performance Trend - {selected_student}",
                            markers=True)
                fig.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
                
                # Subject breakdown
                if len(student_data['Subject'].unique()) > 1:
                    subject_avg = student_data.groupby('Subject')['Percentage'].mean().reset_index()
                    fig2 = px.bar(subject_avg, x='Subject', y='Percentage',
                                title="📚 Average Score by Subject")
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("⚠️ No valid quiz data found.")
        else:
            st.info("📝 No quiz submissions yet. Complete some quizzes to see performance data!")

# ---------------------- GRADIO INTERFACE -----------------------
def create_gradio_interface():
    """Create Gradio interface for the AI Teaching Assistant"""
    
    def gradio_chat(message, history):
        response = ai_teaching_assistant(message)
        history.append([message, response])
        return "", history
    
    def gradio_generate_questions(topic, grade, num_questions):
        return generate_personalized_questions(topic, grade, num_questions)
    
    def gradio_feedback(score, subject, answer):
        return generate_ai_feedback(float(score), subject, answer)
    
    # Create Gradio blocks
    with gr.Blocks(title="AI Education Platform", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎓 AI Education Platform")
        gr.Markdown("*Powered by IBM Granite 3.3-2B Instruct*")
        
        with gr.Tab("🤖 AI Teaching Assistant"):
            chatbot = gr.Chatbot(label="Chat with AI Teacher")
            msg = gr.Textbox(label="Ask a question", placeholder="e.g., Explain photosynthesis...")
            clear = gr.Button("Clear Chat")
            
            msg.submit(gradio_chat, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        
        with gr.Tab("📝 Generate Questions"):
            with gr.Row():
                topic_input = gr.Textbox(label="Topic", placeholder="e.g., Mathematics")
                grade_input = gr.Dropdown(["Elementary", "Middle School", "High School"], label="Grade Level")
                num_input = gr.Slider(1, 5, value=3, label="Number of Questions")
            
            generate_btn = gr.Button("Generate Questions", variant="primary")
            questions_output = gr.Textbox(label="Generated Questions", lines=10)
            
            generate_btn.click(gradio_generate_questions, 
                             [topic_input, grade_input, num_input], 
                             questions_output)
        
        with gr.Tab("💬 AI Feedback"):
            with gr.Row():
                score_input = gr.Slider(0, 100, value=75, label="Score (%)")
                subject_input = gr.Dropdown(["Mathematics", "Science", "English"], label="Subject")
            
            answer_input = gr.Textbox(label="Student Answer (Optional)", lines=3)
            feedback_btn = gr.Button("Generate Feedback", variant="primary")
            feedback_output = gr.Textbox(label="AI Feedback", lines=5)
            
            feedback_btn.click(gradio_feedback,
                             [score_input, subject_input, answer_input],
                             feedback_output)
    
    return demo

# ---------------------- DEPLOYMENT FUNCTIONS -----------------------
def run_streamlit():
    """Run Streamlit app"""
    run_streamlit_app()

def run_gradio():
    """Run Gradio app"""
    # Load model for Gradio
    global granite_generator, tokenizer
    granite_generator, tokenizer = load_granite_model()
    
    demo = create_gradio_interface()
    demo.launch(share=True, debug=True)

# ---------------------- MAIN EXECUTION -----------------------
if __name__ == "__main__":
    import sys
    
    # Check if running in Streamlit
    if "streamlit" in sys.modules:
        run_streamlit_app()
    else:
        # For Google Colab - you can choose which interface to run
        print("Choose interface:")
        print("1. Streamlit")
        print("2. Gradio")
        print("3. Both")
        
        choice = input("Enter choice (1/2/3): ")
        
        if choice == "1":
            # Run Streamlit
            os.system("streamlit run script.py")
        elif choice == "2":
            # Run Gradio
            run_gradio()
        elif choice == "3":
            # Run both (Gradio first, then Streamlit in background)
            thread = threading.Thread(target=run_gradio)
            thread.start()
            time.sleep(5)  # Wait for Gradio to start
            os.system("streamlit run script.py")
