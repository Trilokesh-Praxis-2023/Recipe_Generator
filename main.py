import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # Or try "EleutherAI/gpt-neo-125M"
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
st.title("Recipe Generator with AI")
st.write("Enter a prompt below to generate a recipe!")

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
