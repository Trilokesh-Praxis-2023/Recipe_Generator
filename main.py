import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token to eos_token if not already set
tokenizer.pad_token = tokenizer.eos_token

# Function to generate a detailed recipe
def generate_detailed_recipe(prompt, max_length=400):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        max_length=max_length, 
        do_sample=True, 
        temperature=0.6,  # Slightly lower temperature for coherence
        pad_token_id=tokenizer.eos_token_id
    )
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Formatting the output for readability
    if "Ingredients:" in recipe and "Instructions:" in recipe:
        ingredients, instructions = recipe.split("Instructions:", 1)
        formatted_recipe = (
            f"### Ingredients:\n{ingredients.replace('Ingredients:', '').strip()}\n\n"
            f"### Instructions:\n{instructions.strip()}"
        )
    else:
        formatted_recipe = recipe  # Fallback if the format is unexpected

    return formatted_recipe

# Streamlit UI
st.set_page_config(page_title="AI Recipe Generator", layout="centered")
st.markdown(
    f"""
    <style>
    /* Background Image */
    .stApp {{
        background-image: url('https://img.freepik.com/free-vector/abstract-technology-betwork-wire-mesh-background_1017-17263.jpg');
        background-size: cover;
        background-attachment: fixed;
    }}

    /* Container Styling */
    .stMarkdown, .stButton, .stTextInput, .stSpinner {{
        background: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 10px;
    }}
    
    /* Fonts and Headings */
    h1, h2, h3 {{
        color: #ff6347;
        font-family: 'Arial', sans-serif;
    }}
    p, label {{
        color: #ffffff;
        font-family: 'Verdana', sans-serif;
    }}

    /* Text Input Box */
    .stTextInput {{
        width: 100%;
        padding: 12px;
        border: none;
        border-radius: 5px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }}

    /* Button */
    .stButton>button {{
        background-color: #ff6347;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
    }}

    /* Spinner Styling */
    .stSpinner {{
        color: #ff6347;
        font-weight: bold;
    }}
    </style>
    """, 
    unsafe_allow_html=True
)

st.title("🍲 AI Recipe Generator")
st.write("Enter a prompt below to generate a delicious recipe!")

# User input for prompt
prompt = st.text_input("Recipe Prompt", "Generate a detailed recipe for a classic chicken biryani")

# Button to generate recipe
if st.button("Generate Recipe"):
    if prompt:
        with st.spinner("Generating recipe..."):
            recipe = generate_detailed_recipe(prompt)
            st.markdown(recipe)
    else:
        st.warning("Please enter a prompt.")
