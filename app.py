import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="English to Urdu Translator",
    page_icon="🇵🇰",
    layout="wide"
)

# ---------------------------------------------------
# Model configuration
# ---------------------------------------------------

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ur"


# ---------------------------------------------------
# Load model
# ---------------------------------------------------

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


# Load model
with st.spinner("Loading translation model..."):
    tokenizer, model, device = load_model()


# ---------------------------------------------------
# Translation function
# ---------------------------------------------------

def translate_to_urdu(text):
    """
    Translate English text into Urdu.
    Processes the input paragraph by paragraph.
    """

    if not text or not text.strip():
        return ""

    paragraphs = text.split("\n")

    translated_paragraphs = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            translated_paragraphs.append("")
            continue

        # Tokenize
        inputs = tokenizer(
            paragraph,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move tensors to the same device as the model
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Generate translation
        with torch.no_grad():

            translated_tokens = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

        # Convert tokens to text
        translated = tokenizer.decode(
            translated_tokens[0],
            skip_special_tokens=True
        )

        translated_paragraphs.append(translated)

    return "\n".join(translated_paragraphs)


# ---------------------------------------------------
# Streamlit User Interface
# ---------------------------------------------------

st.title("🇬🇧 English → 🇵🇰 Urdu Translator")

st.write(
    "Enter English text below and translate it into Urdu using AI."
)

st.divider()


# Two-column layout
col1, col2 = st.columns(2)


# ---------------------------------------------------
# English input
# ---------------------------------------------------

with col1:

    st.subheader("🇬🇧 English Text")

    english_text = st.text_area(
        "Enter English text:",
        placeholder="Enter English text here...",
        height=250,
        key="english_input"
    )


# ---------------------------------------------------
# Urdu output
# ---------------------------------------------------

with col2:

    st.subheader("🇵🇰 Urdu Translation")

    # Translation button
    translate_button = st.button(
        "🔄 Translate",
        type="primary",
        use_container_width=True
    )

    if translate_button:

        if not english_text.strip():

            st.warning("Please enter some English text.")

        else:

            with st.spinner("Translating..."):

                try:

                    translation = translate_to_urdu(
                        english_text
                    )

                    st.text_area(
                        "Urdu translation:",
                        value=translation,
                        height=250
                    )

                except Exception as e:

                    st.error(
                        f"An error occurred: {str(e)}"
                    )


# ---------------------------------------------------
# Example sentences
# ---------------------------------------------------

st.divider()

st.subheader("💡 Try an example")

examples = [
    "Hello, how are you?",
    "My name is Ahmed and I live in Pakistan.",
    "Artificial intelligence is changing the world.",
    "I want to learn programming.",
    "Pakistan is a beautiful country."
]

selected_example = st.selectbox(
    "Select an example:",
    ["Select an example..."] + examples
)

if selected_example != "Select an example...":

    st.info(
        f"Example: {selected_example}"
    )

st.divider()

st.caption(
    "Powered by Helsinki-NLP OPUS-MT English → Urdu model"
)