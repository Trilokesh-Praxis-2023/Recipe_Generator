import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
# model_name = "EleutherAI/gpt-neo-1.3B"  # Or try "EleutherAI/gpt-neo-125M"
model_name = "EleutherAI/gpt-neo-125M"  # Or try "EleutherAI/gpt-neo-125M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token to eos_token if not already set
tokenizer.pad_token = tokenizer.eos_token

# Function to generate a more detailed recipe
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
    """
    <style>
    /* Background Image */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1512058564366-c9a2f15f0b06?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwzNjUyOXwwfDF8c2VhcmNofDd8fGZvb2QlMjBiYWNrZ3JvdW5kfGVufDB8fHx8MTY1MjUwNTAyOA&ixlib=rb-1.2.1&q=80&w=1080');
        background-size: cover;
    }

    /* Container Styling */
    .stMarkdown, .stButton, .stTextInput, .stSpinner {
        background: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 10px;
    }
    
    /* Fonts and Headings */
    h1, h2, h3 {
        color: #ff6347; /* Tomato color */
        font-family: 'Arial', sans-serif;
    }
    p, label {
        color: #ffffff;
        font-family: 'Verdana', sans-serif;
    }

    /* Text Input Box */
    .stTextInput {
        width: 100%;
        padding: 12px;
        border: none;
        border-radius: 5px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }

    /* Button */
    .stButton>button {
        background-color: #ff6347;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
    }

    /* Spinner Styling */
    .stSpinner {
        color: #ff6347;
        font-weight: bold;
    }
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
