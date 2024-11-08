import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer (checking the loading process)
model_name = "EleutherAI/gpt-neo-1.3B"

# Loading the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)  # Move the model to GPU if available
    st.success("Model and tokenizer loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model and tokenizer: {str(e)}")

# Set pad_token to eos_token if not already set
tokenizer.pad_token = tokenizer.eos_token

# Function to generate a detailed recipe
def generate_detailed_recipe(prompt, max_length=400):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Move input tensors to the same device as the model (GPU if available)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        max_length=max_length, 
        do_sample=True, 
        temperature=0.6,
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
        formatted_recipe = recipe

    return formatted_recipe

# Embed HTML for background
st.markdown(
    """
    <style>
    body {
        background-image: url("https://cdn.pixabay.com/photo/2024/06/01/14/00/ai-8802304_1280.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp {
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
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
