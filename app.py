import streamlit as st
import random
from transformers import pipeline
import torch
from datetime import date, datetime
import pandas as pd
import os
import json
import plotly.express as px

# ---------------------- UTILS -----------------------
@st.cache_resource
def load_feedback_generator():
    return pipeline("text-generation", model="distilgpt2", device=0 if torch.cuda.is_available() else -1)

def save_quiz_submission(data):
    os.makedirs("quiz_submissions", exist_ok=True)
    file_path = f"quiz_submissions/{data['Name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def rubric_score(answer):
    score = 0
    if len(answer) > 10:
        score += 2
    if any(word in answer.lower() for word in ["because", "therefore", "explains"]):
        score += 2
    if answer[0].isupper() and answer.endswith('.'):
        score += 1
    return min(score, 5)

feedback_generator = load_feedback_generator()

# ---------------------- SIDEBAR NAVIGATION -----------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["🏠 Home", "👤 Student Profile", "📝 Assessments", "💬 Feedback", "📤 Submission", "🧠 Quiz Grader", "📊 Performance Dashboard", "💬 AI Teaching Assistant"])

# ---------------------- AI TEACHING ASSISTANT -----------------------
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")  # Replace with your custom model

qa_model = load_qa_model()

# Sidebar-based chatbot for dynamic Q&A (AI Teaching Assistant)
if section == "💬 AI Teaching Assistant":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Teaching Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.sidebar.text_input("Ask me anything about your subject:")

    if st.sidebar.button("Ask"):
        if user_question:
            st.session_state.chat_history.append(("user", user_question))

            # Constructing context dynamically (you can replace this with real-time subject content)
            context = """
            Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods 
            with the help of chlorophyll. During this process, plants convert carbon dioxide and water into glucose and oxygen.
            """
            
            # Get answer from the Hugging Face QA model
            try:
                result = qa_model(question=user_question, context=context)
                assistant_reply = result["answer"]
            except Exception as e:
                assistant_reply = f"Sorry, I couldn't fetch the answer. Error: {str(e)}"
            
            st.session_state.chat_history.append(("assistant", assistant_reply))

    if st.session_state.chat_history:
        for role, msg in reversed(st.session_state.chat_history[-5:]):
            if role == "user":
                st.sidebar.markdown(f"**You:** {msg}")
            else:
                st.sidebar.markdown(f"**Assistant:** {msg}")

# ---------------------- HOME SECTION -----------------------
elif section == "🏠 Home":
    st.markdown("## **AI-Powered Educational Platform**")
    st.write("Welcome to the AI-powered platform that personalizes learning for students!")

    st.markdown("### **Expected Solutions**")
    st.markdown(""" 
    1. ✅ Increased student engagement and improved academic performance.  
    2. ✅ Reduced workload for teachers through automated grading and content personalization.  
    3. ✅ Enhanced efficiency and effectiveness in educational assessment and instructional planning.
    """)

    st.markdown("### **Technologies & Tools**")
    st.markdown(""" 
    | Component              | Technologies & Tools                          |
    |------------------------|-----------------------------------------------|
    | Generative AI Models   | IBM Granite LLM                               |
    | Programming Language   | Python                                        |
    | Frameworks & Libraries | LangChain, Hugging Face, Pandas, NumPy        |
    | User Interface         | Streamlit                                     |
    | Development Environment| Jupyter Notebook, Google Colab                |
    """, unsafe_allow_html=True)

# ---------------------- STUDENT PROFILE -----------------------
elif section == "👤 Student Profile":
    st.markdown("## **Student Profile**")
    name = st.text_input("Enter student name:")
    age = st.number_input("Enter age:", min_value=5, max_value=100)
    grade = st.selectbox("Select grade:", ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "High School", "College"])
    interests = st.text_area("Learning Interests (e.g., Math, Science, Art):")
    if st.button("Save Profile"):
        st.session_state.student_profile = {"name": name, "age": age, "grade": grade, "interests": interests}
        st.success(f"Profile saved for {name}!")

# ---------------------- ASSESSMENTS -----------------------
elif section == "📝 Assessments":
    st.markdown("## **Assessments**")
    interests = st.session_state.get("student_profile", {}).get("interests", "Math, Science")
    st.write(f"Based on your interests: {interests}")
    topic = st.selectbox("Choose a subject for personalized questions:", interests.split(", "))
    if st.button("Generate Assessment"):
        st.markdown(f"### Sample Questions for **{topic}**")
        questions = {
            "Math": ["What is 12 x 8?", "Solve for x: 3x + 5 = 20"],
            "Science": ["What is H2O?", "Explain photosynthesis."],
            "English": ["Use 'courage' in a sentence.", "What is a simile?"]
        }
        for q in random.sample(questions.get(topic, ["Define your topic."]), 2):
            st.write("- " + q)

# ---------------------- FEEDBACK -----------------------
elif section == "💬 Feedback":
    st.markdown("## **Feedback**")
    score = st.slider("Student's score (%)", 0, 100, 70)
    performance = "excellent" if score >= 85 else "moderate" if score >= 60 else "needs improvement"
    prompt = f"Give feedback for a student who scored {score}% and has {performance} performance."
    if st.button("Generate Feedback"):
        with st.spinner("Generating feedback..."):
            result = feedback_generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            st.write(result)

# ---------------------- SUBMISSION -----------------------
elif section == "📤 Submission":
    st.markdown("## 📚 Project Submission Portal")
    if "submission_start" not in st.session_state:
        st.session_state.submission_start = date.today()
    if "submission_end" not in st.session_state:
        st.session_state.submission_end = date.today()
    st.sidebar.markdown("### Admin Panel")
    if st.sidebar.checkbox("I'm an admin"):
        st.session_state.submission_start = st.sidebar.date_input("Submission Start Date", st.session_state.submission_start)
        st.session_state.submission_end = st.sidebar.date_input("Submission End Date", st.session_state.submission_end)

    today = date.today()
    start, end = st.session_state.submission_start, st.session_state.submission_end
    status = "🟢 Open" if start <= today <= end else "🔴 Closed"
    st.markdown(f"### Submission Status: {status}")

    if status == "🟢 Open":
        with st.form("submission_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            subject = st.text_input("Subject")
            uploaded_file = st.file_uploader("Upload file", type=["pdf", "zip"])
            submitted = st.form_submit_button("Submit")
            if submitted and name and email and subject and uploaded_file:
                df = pd.DataFrame([{"Name": name, "Email": email, "Subject": subject, "File": uploaded_file.name, "Date": str(date.today())}])
                os.makedirs("uploads", exist_ok=True)
                with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.read())
                df.to_csv("submissions.csv", mode="a", index=False, header=not os.path.exists("submissions.csv"))
                st.success("Submission successful!")
            elif submitted:
                st.error("Please fill all fields.")

# ---------------------- AI QUIZ GRADER -----------------------
elif section == "🧠 Quiz Grader":
    st.markdown("## 🧠 AI-Powered Quiz Grader")
    with st.form("quiz_form"):
        name = st.text_input("Student Name")
        subject = st.selectbox("Subject", ["Math", "Science", "English"])
        mcq_q = st.radio("What is the capital of France?", ["Berlin", "Madrid", "Paris", "Rome"])
        short_answer = st.text_area("Explain the process of photosynthesis:")
        uploaded_text = st.file_uploader("Upload written answers (TXT)", type=["txt"])
        submit_quiz = st.form_submit_button("Grade Quiz")

        if submit_quiz:
            mcq_score = 5 if mcq_q == "Paris" else 0
            rubric = rubric_score(short_answer)
            file_answer = ""
            file_score = 0
            if uploaded_text:
                content = uploaded_text.read().decode("utf-8")
                file_answer = content
                file_score = rubric_score(content)

            total = mcq_score + rubric + file_score
            feedback = feedback_generator(f"Give feedback for a student who scored {total} out of 15.", max_length=80)[0]['generated_text']
            st.markdown("### 🧾 Results")
            st.write(f"- MCQ Score: {mcq_score}/5")
            st.write(f"- Written Score: {rubric}/5")
            st.write(f"- Uploaded Answer Score: {file_score}/5")
            st.write(f"### 💬 Feedback: {feedback}")

            save_quiz_submission({
                "Name": name,
                "Subject": subject,
                "MCQ": mcq_q,
                "Short Answer": short_answer,
                "Short Score": rubric,
                "File Answer": file_answer,
                "File Score": file_score,
                "MCQ Score": mcq_score,
                "Total": total,
                "Date": str(datetime.now())
            })

# ---------------------- PERFORMANCE DASHBOARD -----------------------
elif section == "📊 Performance Dashboard":
    st.markdown("## 📊 Student Performance Dashboard")
    if os.path.exists("quiz_submissions"):
        all_files = os.listdir("quiz_submissions")
        data = []
        for file in all_files:
            with open(f"quiz_submissions/{file}", "r") as f:
                data.append(json.load(f))

        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])

        student_names = df['Name'].unique()
        selected_student = st.selectbox("Select a student", student_names)

        filtered = df[df['Name'] == selected_student].sort_values("Date")

        fig = px.line(filtered, x="Date", y="Total", title=f"Performance of {selected_student} Over Time", markers=True)
        st.plotly_chart(fig)
    else:
        st.warning("No quiz data found yet. Grade some quizzes to populate the dashboard.")
