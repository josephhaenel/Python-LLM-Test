from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from transformers.utils import logging
from typing import Optional

logging.set_verbosity_info()

# Load configuration from a file or environment variables
CONFIG = {
    "model_name": "./fine-tuned-model",  # Use the fine-tuned model
    "max_length": 1000,
    "padding_side": "left"
}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], padding_side=CONFIG["padding_side"])
model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"])

def clear_terminal():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def encode_input(user_input: str) -> torch.Tensor:
    """Encode user input with the tokenizer."""
    return tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

def generate_response(bot_input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
    """Generate a response from the model."""
    return model.generate(bot_input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

def decode_response(chat_history_ids: torch.Tensor, bot_input_ids: torch.Tensor) -> str:
    """Decode the response from the model."""
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

def print_help():
    """Print the help message."""
    print('!reset - Resets the conversation and clears the terminal screen.')
    print('!exit - Exits the chatbot.')

def preprocess_input(user_input: str) -> str:
    """Preprocess the user input."""
    return user_input.lower().strip()

def chatbot_response(user_input: str, chat_history_ids: Optional[torch.Tensor]) -> str:
    """Generate a chatbot response."""
    new_user_input_ids = encode_input(user_input)
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = generate_response(bot_input_ids, CONFIG["max_length"])
    model_response = decode_response(chat_history_ids, bot_input_ids)
    return model_response, chat_history_ids

def main():
    """Main function to run the chatbot."""
    try:
        clear_terminal()
        print("Chatbot started. Type your message below.\nType '!help' for a list of commands.")

        chat_history_ids: Optional[torch.Tensor] = None

        while True:
            user_input = input('You: ')
            user_input = preprocess_input(user_input)

            if user_input == '!help':
                print_help()
                continue

            if user_input == '!reset':
                chat_history_ids = None
                clear_terminal()
                print("Conversation reset. Type your message below.\nType '!help' for a list of commands.")
                continue

            if user_input == '!exit':
                print("Ending the conversation. Goodbye!")
                break

            try:
                bot_response, chat_history_ids = chatbot_response(user_input, chat_history_ids)
                print(f'DialoGPT: {bot_response}')
            except Exception as e:
                print(f"An error occurred: {e}")
                logging.error("Error during chatbot interaction", exc_info=True)

    except Exception as e:
        print(f"Failed to start the chatbot: {e}")
        logging.error("Failed to start the chatbot", exc_info=True)

if __name__ == "__main__":
    main()
