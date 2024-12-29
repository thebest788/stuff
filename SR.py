from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sys

# Load GPT-2 model and tokenizer
model_name = "gpt2-xl"  # You can change this to another model if needed
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model.eval()  # Set the model to evaluation mode

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.9, temperature=1.0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    user_input = sys.argv[1]  # Get input text from command line
    print(generate_response(user_input))
