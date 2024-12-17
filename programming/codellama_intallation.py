from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "codellama/CodeLlama-13b-Python-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set eos_token as pad_token
tokenizer.pad_token = tokenizer.eos_token

# Using quantization for memory efficiency
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config  # Updated from load_in_4bit=True
)

# Function to generate text
def generate_text(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Pass the attention mask
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = '''	
def choose_num(x, y): """This function takes two positive numbers x and y and returns the biggest even integer number that is in the range [x, y] inclusive. If there's no such number, then the function should return -1. For example: choose_num(12, 15) = 14 choose_num(13, 12) = -1 """
'''
generated_text = generate_text(prompt, max_length=512, temperature=0.3)
print(generated_text)


