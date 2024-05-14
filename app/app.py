from fastapi import FastAPI
from transformers import GemmaTokenizer, AutoModelForCausalLM
import torch
from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str

app = FastAPI()

# Load the fine-tuned language model and tokenizer
gpu = torch.device('cuda:0')
model_id = "google/codegemma-2b"
tokenizer = GemmaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=gpu, torch_dtype=torch.float16)

@app.post("/highlight-code/")
async def highlight_code(prompt: Prompt):
    print(prompt)
    inputs = tokenizer(prompt.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]

    differences = []

    prompt_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    result = []
    last_dif = 0
    for j, i in enumerate(logits):
        probabilities = torch.softmax(i, 0)
        actual_token = torch.zeros([1], dtype=torch.int32)
        if j + 1 < len(inputs["input_ids"][0]):
            actual_token = inputs["input_ids"][0][j+1]
        act_prob = probabilities[actual_token]
        pred_prob = torch.max(probabilities)

        difference =  pred_prob.item() - act_prob.item()
        differences.append(difference)

        token = prompt_tokens[j]

        threshold = 0.2
        if last_dif <= threshold:
            token = token.replace("▁", " ")
        result.append((last_dif, token))
        last_dif = difference

    return {"generated_text": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
