from flask import Flask, request, jsonify
import threading
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

app = Flask(__name__)

def load_model():
    adapter_path = "adapter_model"
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = "bagelnet/Llama-3-8B"

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer

model, tokenizer = load_model()

@app.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello, World!'

@app.route('/generate', methods=['POST'])
def generate(max_length=300):
  data = request.get_json()
  clinical_note_summary = data.get('clinical note summary', '')

  if not clinical_note_summary:
      return jsonify({'error': 'No conversation provided'}), 400
  else:
      print("Provided Clinical note summary:", clinical_note_summary)

  # Prepare the prompt
  prompt = f"""
    You are an expert medical coding assistant.

    Task: Analyze the following summary of a clinical note and provide a list of appropriate ICD-10-CM codes that best relate to the medical information mentioned.

    Instructions:
    - Provide a maximum of 4 ICD-10-CM codes.
    - Format: '[Code]: [Description]'
    - List each code and its description on a new line.
    - Only include the codes and their descriptions.
    - Do NOT include any other text or commentary.
    - Take a deep breath before you answer.


  Clinical Note Summary:
  {clinical_note_summary}
  """

  # Encode the input prompt
  input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

  # Generate a response
  with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + max_length,
        num_return_sequences=1,
        # no_repeat_ngram_size=2,
        temperature=1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

  # Decode and return the response
  response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
  response_text = response.strip()

  return jsonify({'response': response_text})

@app.route('/shutdown', methods=['GET'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        print('Could not find the server shutdown function.')
        return 'Server shutdown failed.'
    else:
        print('Shutting down server...')
        func()
        return 'Server shutting down...'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)