# Importing the necessary libraries and loading the pre-trained model and tokenizer for the chatbot.
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
import torch

chat_history_ids = None

def chat(user_input):
    '''
    This function takes in a user input and returns a response.

    Args:
        user_input (str): The user's input.

    Returns:
        str: The chatbot's response.
    '''
    global chat_history_ids
    # Encode the user input and append it to the chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Concatenate the new input with the chat history (if it exists) and generate a response from the model
    bot_input_ids = (torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids)
    chat_history_ids = model.generate(bot_input_ids,
                                      max_length=1000,
                                      pad_token_id=tokenizer.eos_token_id
                                      )

    # Decode the generated response and return it as a string
    response = tokenizer.decode(chat_history_ids[:,bot_input_ids.shape[-1]:][0],
                                skip_special_tokens=True
                                )
    return response


# Start a conversation with the chatbot. The user can type "exit" to end the conversation.
while True:
    user_input = input(">> User: ")
    if user_input.lower() == "exit":
        break
    print(">> BotKay: ", chat(user_input))