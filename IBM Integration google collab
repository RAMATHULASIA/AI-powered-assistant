
# Cell 1: Install all required packages
!pip install -q streamlit gradio transformers torch accelerate plotly pandas pyngrok
!pip install -q --upgrade huggingface_hub

# Verify installation
import streamlit as st
import gradio as gr
import transformers
print("✅ All packages installed successfully!")


# Cell 2: Test IBM Granite Model Loading
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def test_model():
    print("🧠 Loading IBM Granite 3.3-2B Instruct...")

    model_name = "ibm-granite/granite-3.3-2b-instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    # Create pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # Test the model
    test_prompt = "Explain photosynthesis for a high school student:"
    result = generator(test_prompt, max_new_tokens=100, temperature=0.7)

    print("✅ Model loaded successfully!")
    print(f"Sample output: {result[0]['generated_text']}")

    return generator, tokenizer

# Run the test
generator, tokenizer = test_model()



# Cell 3: Quick Gradio Deployment
import gradio as gr

def create_education_app(generator):

    def generate_questions(topic, grade_level, num_questions):
        prompt = f"""Generate {num_questions} educational questions for {grade_level} students about {topic}.
        Make them engaging and appropriate for the grade level. Format as a numbered list:

        Topic: {topic}
        Grade: {grade_level}
        Questions:"""

        result = generator(prompt, max_new_tokens=200, temperature=0.7)
        return result[0]['generated_text']

    def generate_feedback(score, subject, student_answer):
        performance_level = "excellent" if score >= 85 else "good" if score >= 70 else "needs improvement"

        prompt = f"""As an AI teaching assistant, provide constructive feedback for a student who scored {score}% in {subject}.
        Their performance level is {performance_level}.
        Student's answer: "{student_answer}"

        Provide encouraging and specific feedback:"""

        result = generator(prompt, max_new_tokens=150, temperature=0.7)
        return result[0]['generated_text'].split("Provide encouraging and specific feedback:")[-1].strip()

    def chat_with_ai(message, history):
        prompt = f"""You are an AI teaching assistant. Answer the student's question clearly and helpfully.

        Student Question: {message}

        Answer:"""

        result = generator(prompt, max_new_tokens=200, temperature=0.7)
        response = result[0]['generated_text'].split("Answer:")[-1].strip()

        history.append([message, response])
        return "", history

    # Create Gradio interface
    with gr.Blocks(title="AI Education Platform", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎓 AI Education Platform")
        gr.Markdown("### *Powered by IBM Granite 3.3-2B Instruct*")

        with gr.Tab("🤖 AI Teaching Assistant"):
            gr.Markdown("Ask me anything about your studies!")
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                label="Your Question",
                placeholder="e.g., Explain the water cycle, Help with algebra..."
            )
            clear = gr.Button("Clear Chat")

            msg.submit(chat_with_ai, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

        with gr.Tab("📝 Question Generator"):
            gr.Markdown("Generate personalized educational questions")

            with gr.Row():
                topic_input = gr.Textbox(
                    label="Subject/Topic",
                    placeholder="e.g., Mathematics, Biology, History"
                )
                grade_input = gr.Dropdown(
                    ["Elementary School", "Middle School", "High School", "College"],
                    label="Grade Level",
                    value="High School"
                )
                num_input = gr.Slider(1, 5, value=3, label="Number of Questions")

            generate_btn = gr.Button("🎯 Generate Questions", variant="primary")
            questions_output = gr.Textbox(
                label="Generated Questions",
                lines=10,
                placeholder="Generated questions will appear here..."
            )

            generate_btn.click(
                generate_questions,
                [topic_input, grade_input, num_input],
                questions_output
            )

        with gr.Tab("💬 AI Feedback Generator"):
            gr.Markdown("Get personalized feedback on student performance")

            with gr.Row():
                score_input = gr.Slider(0, 100, value=75, label="Student Score (%)")
                subject_input = gr.Dropdown(
                    ["Mathematics", "Science", "English", "History", "Art"],
                    label="Subject",
                    value="Mathematics"
                )
        answer_input = gr.Textbox(
                label="Student's Answer (Optional)",
                lines=4,
                placeholder="Paste the student's response here for more personalized feedback..."
            )

            feedback_btn = gr.Button("🎯 Generate Feedback", variant="primary")
            feedback_output = gr.Textbox(
                label="AI-Generated Feedback",
                lines=6,
                placeholder="Personalized feedback will appear here..."
            )

            feedback_btn.click(
                generate_feedback,
                [score_input, subject_input, answer_input],
                feedback_output
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Platform

            This AI Education Platform uses **IBM Granite 3.3-2B Instruct** model to provide:

            - 🤖 **Intelligent Teaching Assistant**: Get instant help with any academic question
            - 📝 **Personalized Question Generation**: Create custom assessments for any topic
            - 💬 **AI-Powered Feedback**: Receive detailed, constructive feedback on performance

            ### How to Use:
            1. **AI Teaching Assistant**: Ask questions in natural language
            2. **Question Generator**: Enter topic and grade level to generate questions
            3. **Feedback Generator**: Input score and subject for personalized feedback

            ### Technologies Used:
            - **AI Model**: IBM Granite 3.3-2B Instruct
            - **Framework**: Gradio for web interface
            - **Platform**: Google Colab
            - **Libraries**: Transformers, PyTorch
            """)

    return demo

# Launch the application
print("🚀 Launching AI Education Platform...")
demo = create_education_app(generator)
demo.launch(share=True, debug=True)
