from fastapi import FastAPI
from transformers import GemmaTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import BaseModel
import json
import re


class Prompt(BaseModel):
    prompt: str

app = FastAPI()

# Load the fine-tuned language model and tokenizer
gpu = torch.device('cuda:0')
model_id = "google/codegemma-1.1-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=gpu, torch_dtype=torch.bfloat16)
new_prompt = '''
Objective: Identify lines of code that might contain bugs.

Context: I have a codebase written in c++. I need to find lines that might contain bugs based on specific criteria.

Criteria for Identifying Potential Bugs:

    1. **Syntax Errors:** Look for lines with syntax errors.
    2. **Logical Errors:** Identify lines where the logic might be flawed (e.g., incorrect conditions, improper use of operators).
    3. **Runtime Errors:** Find lines that might cause runtime errors (e.g., null pointer dereferences, out-of-bound array access).
    4. **Common Pitfalls:** Check for common pitfalls in the language (e.g., off-by-one errors, improper resource management).
    5. **Code Quality Issues:** Look for lines that deviate from standard coding practices (e.g., poor variable naming, lack of comments, complex expressions).


Instructions:

    1. Analyze the provided codebase and identify lines that match the above criteria.
    2. Make sure to check for edge cases and scenarios where the input might cause unexpected behavior.
    3. Pay attention to lines with complex logic or multiple operations as they are more prone to errors.
    4. For each identified line, provide a json object with the following fields:
        - problematic_line_of_code: The line of code as it is in the original source code. 
        - description: A description of the potential bug.
        - suggestion: A suggestion for how to fix or further investigate the issue.

**Important**:
- Use exact substrings from the original code in the "problematic_line_of_code" field to ensure they can be matched using the Python `find` method.
- Make sure each "description" and "suggestion" is relevant to the identified line of code.
- Do not output anything but the json objects for each identified line of code.

Here is an example of how the code could look like and what your response should be:

Example1 - Code:
```c++
double calculate_average(std::vector<int> numbers) {
    int total = std::accumulate(numbers.begin(), numbers.end(), 0);
    int count = numbers.size();
    return total / static_cast<double>(count);
}

int get_element(std::vector<int> array, int index) {
    return array.at(index);
}
```

Example1 - Response:

```json
[{  problematic_line_of_code: "return total / static_cast<double>(count);",
    description: "Potential division by zero if numbers vector is empty",
    suggestion: "Add a check to ensure count is not zero before performing the division." },
    { problematic_line_of_code: "return array.at(index);",
    description: "Possible out-of-bound array access",
    suggestion: "Add a check to ensure the index is within the bounds of the array." }]
```

Example2 - Code:

```c++
int calculate_sum(std::vector<int> numbers) {
    int sum = 3;
    for (int j = 0; j <= numbers.size(); j++) {
        sum += numbers[j];
    }
    return sum;
}
```

Example 2 - Response

```json
[{  problematic_line_of_code: "for (int j = 0; j <= numbers.size(); j++) {",
    description: "The loop condition is incorrect because j is used as index to numbers. It should be j < numbers.size() instead of j <= numbers.size().",
    suggestion: "Change the loop condition to j < numbers.size() to prevent out of bounds accesses to the vector."},
    { problematic_line_of_code: "int sum = 3;",
    description: "It is unusual that the sum variable is initialized to a non-zero value.",
    suggestion: "Change the line to int sum = 0;"}]
```

Example3 - Code:

```c++
int calculate_sum(std::vector<int> numbers) {
    int sum = 0;
    for (int j = 0; j < numbers.size(); j++) {
        sum -= numbers[j];
    }
    return sum;
}
```

Example 3 - Response

```json
[{  problematic_line_of_code: "sum -= numbers[j];",
    description: "The operator -= is wrong to calculate a sum. It should be +=.",
    suggestion: "Change the operator to +=."}]
```
    


Please review the code below and use the formatting of the example to provide your response.

Code to analyze:

```c++
'''

max_new_tokens = 1000
context_window_size = 8000
json_extraction_re = re.compile(r"```json\n(.*)\n```", re.DOTALL)

def recursive_prompting(counter, chat, prompt):
    print(f"Starting iteration: {counter}")
    if counter > 10:
        print("Recursion depth exceeded")
        return []
    with torch.no_grad():
        chat_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]
    if prompt_len > context_window_size - max_new_tokens:
        print("Context size exceeded!")
        return []
    print(f"Prompting with context length: {prompt_len}")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    result = tokenizer.decode(outputs[0][prompt_len:])

    del inputs
    del outputs
    torch.cuda.empty_cache()

    processed_result = re.match(json_extraction_re, result).group(1)
    print(f"Model generated: {len(result)} characters")
    print(f"Extracted JSON: {processed_result}")
    result_json = []
    try:
        result_json = json.loads(processed_result)
    except:
        print(f"Json parsing failed: {processed_result}")
        chat.append({"role" : "assistant", "content": result})
        chat.append({"role" : "user", "content" : """
It looks like the last output provided is not in the correct JSON format. Please ensure that the response is a JSON array of objects, where each object contains the fields "problematic_line_of_code", "description", and "suggestion". The JSON format should look like this:

```json
[
  {
    "problematic_line_of_code": "<line of code>",
    "description": "<description of the potential bug>",
    "suggestion": "<suggestion for fixing or investigating the issue>"
  },
  ...
]
```

Please try again with the code provided earlier, and ensure the output is formatted as valid JSON.
"""})
        return recursive_prompting(counter + 1, chat, prompt)
    for finding in result_json:
        line_of_code = finding["problematic_line_of_code"]
        first_index = prompt.find(line_of_code)
        second_index = prompt.find(line_of_code, first_index +1)
        finding["line_number"] = -1
        if second_index != -1:
            print("Line of code is not unique")
            chat.append({"role" : "assistant", "content" : result})
            chat.append({"role" : "user", "content" : f"""
It seems that the problematic line of code of this json object is ambiguous within the code:
```json
{json.dumps(finding)}
```

Please make sure to clearly specify the problematic line by using an exact substring from the original code. This will help ensure that the identified lines are unambiguous and directly match the provided code.

Please review the code provided earlier and generate the output again, ensuring that each "problematic_line_of_code" is an exact substring from the original code.
"""})
            return recursive_prompting(counter + 1, chat, prompt)
        if first_index == -1:
            print("Line of code could not be found")
            chat.append({"role" : "assistant", "content" : result})
            chat.append({"role" : "user", "content" : f"""
It appears that the identified lines of code in this json object does not exist in the original code provided:
```json
{json.dumps(finding)}
```
                         
Please ensure that the "problematic_line_of_code" field contains exact lines or substrings that are present in the original code.

Please analyze the code again and make sure that each identified "problematic_line_of_code" corresponds exactly to a line or substring from the original source code.
"""})
            return recursive_prompting(counter + 1, chat, prompt)
        finding["line_number"] = prompt[:first_index].count("\n") + 1
    return result_json


@app.post("/highlight-code/")
async def highlight_code(prompt: Prompt):
    chat = [
        { "role": "user", "content": new_prompt + prompt.prompt + "\n```"},
    ]
    print(f"Handling prompt: {prompt.prompt}")
    result = recursive_prompting(0, chat, prompt.prompt)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
