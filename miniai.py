import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Mini AI Toolkit",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini AI Toolkit")
st.write("Powered by Hugging Face Transformers")

# -----------------------------
# Load Models (Cached)
# -----------------------------

@st.cache_resource
def load_text_generator():
    return pipeline(
        "text-generation",
        model="distilgpt2"
    )

@st.cache_resource
def load_sentiment():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_ner():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

@st.cache_resource
def load_qa():
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

# -----------------------------
# Sidebar
# -----------------------------

task = st.sidebar.selectbox(
    "Select NLP Task",
    (
        "Text Generation",
        "Sentiment Analysis",
        "Named Entity Recognition",
        "Question Answering"
    )
)

# ======================================================
# Text Generation
# ======================================================

if task == "Text Generation":

    st.header("📝 Text Generation")

    generator = load_text_generator()

    prompt = st.text_area(
        "Enter Prompt",
        "Artificial Intelligence is"
    )

    length = st.slider(
        "Maximum Tokens",
        20,
        150,
        60
    )

    if st.button("Generate"):

        with st.spinner("Generating..."):

            output = generator(
                prompt,
                max_length=length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8
            )

            st.success("Generated Text")

            st.write(output[0]["generated_text"])

# ======================================================
# Sentiment Analysis
# ======================================================

elif task == "Sentiment Analysis":

    st.header("😊 Sentiment Analysis")

    sentiment = load_sentiment()

    text = st.text_area(
        "Enter Text",
        "I really love Hugging Face!"
    )

    if st.button("Analyze"):

        result = sentiment(text)

        st.success("Prediction")

        st.write("Label :", result[0]["label"])
        st.write("Confidence :", round(result[0]["score"],4))

# ======================================================
# NER
# ======================================================

elif task == "Named Entity Recognition":

    st.header("🏷️ Named Entity Recognition")

    ner = load_ner()

    text = st.text_area(
        "Enter Text",
        "Elon Musk is the CEO of Tesla located in Texas."
    )

    if st.button("Extract Entities"):

        result = ner(text)

        st.success("Entities Found")

        for entity in result:

            st.write(f"""
Entity : {entity['word']}

Type : {entity['entity_group']}

Confidence : {round(entity['score'],4)}

----------------------------
""")

# ======================================================
# Question Answering
# ======================================================

else:

    st.header("❓ Question Answering")

    qa = load_qa()

    context = st.text_area(
        "Context",
        """Streamlit is an open-source Python library
        used to build interactive web applications
        for machine learning and data science."""
    )

    question = st.text_input(
        "Question",
        "What is Streamlit?"
    )

    if st.button("Get Answer"):

        result = qa(
            question=question,
            context=context
        )

        st.success("Answer")

        st.write(result["answer"])

        st.write("Confidence :", round(result["score"],4))