# CosmoSpeak

<center>
    <img src="https://i.postimg.cc/mrJvQpkL/speak.png" alt="CosmoSpeak" width="1216" height="832">
</center>

## Model Summary

CosmoSpeak is a state-of-the-art chatbot that specializes in the domain of Astronautics / Space Mission Engineering. It covers topics such as 

.Flight control team
.Flight Dynamics
.Procedure Preparation and Validation
.Mission Planning
.Extravehicular Activities (EVAs)
.Collision Avoidance Manoeuvres
.Mission Termination and De-Orbit Strategies



CosmoSpeak is a fine-tuned SmolLM-135M trained with Astrochat dataset (https://huggingface.co/datasets/patrickfleith/AstroChat) 

# Model Trained Using AutoTrain

This model was trained using AutoTrain. For more information, please visit [AutoTrain](https://hf.co/docs/autotrain).

# Usage

```python

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "https://huggingface.co/yd915/CosmoSpeak"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# Prompt content: "hi"
messages = [
    {"role": "user", "content": "hi"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)
```
