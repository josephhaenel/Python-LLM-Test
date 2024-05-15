from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from transformers.utils import logging

logging.set_verbosity_info()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", padding_size="left")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Function to clear the terminal screen
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# Main function to run the chatbot
def chatbot():
    # Clear the terminal screen
    clear_terminal()

    # Opening message to user
    print("Chatbot started. Type your message below.\nType '!help' for a list of commands.")

    # Initialize the chat history
    chat_history_ids = None

    # Loop to keep the chatbot running
    while True:
        # Wait for user input
        user_input = input('You: ')

        if user_input.lower() == '!help':
            print('!reset - Resets the conversation and clears the terminal screen.')
            print('!exit - Exits the chatbot.')
            continue  # Continue the loop after showing help

        # Check for commands
        if user_input.lower() == '!reset':
            chat_history_ids = None
            clear_terminal()
            print("Conversation reset. Type your message below.\nType '!help' for a list of commands.")
            continue
        elif user_input.lower() == '!exit':
            print("Ending the conversation. Goodbye!")
            break

        # Encode the new user input, add the eos_token and return a tensor in PyTorch
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

        # Generate a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Pretty print the last output tokens from bot
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f'DialoGPT: {bot_response}')

if __name__ == "__main__":
    chatbot()
