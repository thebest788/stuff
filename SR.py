from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the GPT-2 model and tokenizer
model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model.eval()  # Set the model to evaluation mode

def generate_response(input_text):
    # Encode input text and generate response
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        # Generate response with parameters adjusted for better code generation
        outputs = model.generate(
            input_ids, 
            max_length=300,  # Increase max length for better responses
            num_return_sequences=1, 
            no_repeat_ngram_size=2, 
            top_p=0.9, 
            temperature=0.7,  # Adjusted for better responses
            pad_token_id=tokenizer.eos_token_id  # Avoid padding issues
        )

    # Decode and return the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("AI Chatbot (GPT-2-XL): Ask me anything. Type 'exit' to stop.")
    
    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        ai_response = generate_response(user_input)
        
        if ai_response.strip() == "":
            print("AI couldn't generate a response, please try again.")
        else:
            print(f"AI: {ai_response}")

if __name__ == "__main__":
    main()
