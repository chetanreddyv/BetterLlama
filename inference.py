# Loads my trained model from hugging face 
from unsloth import FastLanguageModel
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Chetanreddy1/Llama-1B-CoT", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = False,
    )
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "2 + 2"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

# Explicitly set attention mask
attention_mask = inputs.ne(tokenizer.pad_token_id).long()

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=inputs,
    attention_mask=attention_mask,  # Add attention mask here
    streamer=text_streamer,
    max_new_tokens=200,
    use_cache=True,
    temperature=1.5,
    min_p=0.1,
)
