import streamlit as st
from transformers import pipeline

st.title("🤖 Mini AI Toolkit")

# Load models
@st.cache_resource
def load_models():
    sentiment = pipeline(
        "sentiment-analysis"
    )

    qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad")

    print("Working")

    generator = pipeline("text-generation")

    ner = pipeline("ner")

    return sentiment, qa, generator, ner

sentiment, qa, generator, ner = load_models()

option = st.sidebar.selectbox(
    "Select Task",
    [
        "Sentiment Analysis",
        "Question Answering",
        "Text Generation",
        "NER"
    ]
)

# Sentiment Analysis
if option == "Sentiment Analysis":

    st.header("Sentiment Analysis")

    text = st.text_area("Enter Text")

    if st.button("Analyze Sentiment"):

        result = sentiment(text)

        st.write("Result:")
        st.write(result)

# Question Answering
elif option == "Question Answering":

    st.header("Question Answering")

    context = st.text_area("Enter Context")

    question = st.text_input("Enter Question")

    if st.button("Get Answer"):

        result = qa(
            question=question,
            context=context
        )

        st.success(
            f"Answer: {result['answer']}"
        )

        st.write(
            f"Confidence: {result['score']:.4f}"
        )

# Text Generation
elif option == "Text Generation":

    st.header("GPT-2 Text Generation")

    prompt = st.text_area("Enter Prompt")

    if st.button("Generate"):

        result = generator(
            prompt,
            max_length=100,
            num_return_sequences=1
        )

        st.write(
            result[0]["generated_text"]
        )

# NER
elif option == "NER":

    st.header("Named Entity Recognition")

    text = st.text_area("Enter Text")

    if st.button("Extract Entities"):

        entities = ner(text)

        for entity in entities:

            st.write(
                f"{entity['word']} → {entity['entity_group']}"
            )