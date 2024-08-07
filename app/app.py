from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import BaseModel
import json
import re
import copy


class Prompt(BaseModel):
    code: str
    language: str
    context: str = ""
    debugState: str = ""


app = FastAPI()

# Load the fine-tuned language model and tokenizer
gpu = torch.device("cuda:0")
model_id = "google/codegemma-1.1-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map=gpu, torch_dtype=torch.bfloat16
)
request_counter = 0
new_prompt = """
# Task description
Objective: Identify lines of code that are relevant for fixing bugs.

Context: I have a codebase in which I suspect bug and I need to find lines that are relevant for fixing this bug.

Here are some criteria for identifying those lines:

    1. **Logical Errors:** Identify lines where the logic might be flawed (e.g., incorrect conditions, improper use of operators).
    2. **Runtime Errors:** Find lines that might cause runtime errors (e.g., null pointer dereferences, out-of-bound array access).
    3. **Common Pitfalls:** Check for common pitfalls in the language (e.g., off-by-one errors, improper resource management).
    4. **Code Quality Issues:** Look for lines that deviate from standard coding practices (e.g., poor variable naming, lack of comments, complex expressions).
    5. **Missing Code:** Identify lines before or after which additional code should be inserted to fix the bug.

Instructions:

    1. Analyze the provided codebase and identify lines that match the above criteria.
    2. Make sure to check for edge cases and scenarios where the input might cause unexpected behavior.
    3. Pay attention to lines with complex logic or multiple operations as they are more prone to errors.
    4. For each identified line, provide a json object with the following fields:
        - problematic_line_of_code: A substring of the original source code that is relevant for fixing the bug and is unique within the source code. 
        - description: A description why this line is relevant and how it contributes to the bug.
        - suggestion: A suggestion for how to fix or further investigate the issue.
        - action: Indicates what action should be taken. Possible values are 'change', 'insert_above', or 'insert_below'. Use 'change' if the problematic line itself needs to be modified. Use 'insert_above' if another line of code is missing before the problematic line. Use 'insert_below' if another line of code is missing after the problematic line.

**Important**:
- Use exact substrings from the original code in the "problematic_line_of_code" field to ensure they can be unambiguously matched using the Python `find` method.
- Make sure each "description" and "suggestion" is relevant to the identified line of code.
- Do not output anything but the json objects for each identified line of code.

# Examples
Here are examples of how the code could look like and what your response should be:

## Example1

Code:
```python
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / float(count)

def get_element(array, index):
    result = array[index]
```

Response:
```json
[
    {
        "problematic_line_of_code": "return total / float(count)",
        "description": "Potential division by zero if numbers list is empty",
        "suggestion": "Add a check to ensure count is not zero before performing the division.",
        "action": "insert_above"
    },
    {
        "problematic_line_of_code": "result = array[index]",
        "description": "Possible out-of-bound list access",
        "suggestion": "Add a check to ensure the index is within the bounds of the list.",
        "action": "insert_above"
    },
    {
        "problematic_line_of_code": "result = array[index]",
        "description": "Missing return statement.",
        "suggestion": "Add a return statement in the next line. The function should return the result.",
        "action": "insert_below"
    },
    
]
```

## Example2

Code:
```cpp
int calculate_sum(std::vector<int> numbers) {
    int sum = 3;
    for (int j = 0; j <= numbers.size(); j++) {
        sum += numbers[j];
    }
    return sum;
}

int calculate_product(std::vector<int> numbers) {
    int product = 1;
    for (int j = 0; j < numbers.size(); j++) {
        product /= numbers[i];
    }
    return product;
}
```

Response:
```json
[{  "problematic_line_of_code" : "for (int j = 0; j <= numbers.size(); j++) {",
    "description" : "The loop condition is incorrect because j is used as index to numbers. It should be j < numbers.size() instead of j <= numbers.size().",
    "suggestion" : "Change the loop condition to j < numbers.size() to prevent out of bounds accesses to the vector.",
    "action": "change"},
    { "problematic_line_of_code" : "int sum = 3;",
    "description" : "It is unusual that the sum variable is initialized to a non-zero value.",
    "suggestion" : "Change the line to int sum = 0;",
    "action": "change"},
    { "problematic_line_of_code" : "product /= numbers[i];",
    "description" : "The operator /= is wrong to calculate a product. It should be *=.",
    "suggestion" : "Change the operator to *=.",
    "action": "change"}]
```

## Example3

Code:
```java
import java.util.HashMap;

public class MapOperations {
    public int getValue(HashMap<String, Integer> map, String key) {
        return map.get(key);
    }

    public void incrementValue(HashMap<String, Integer> map, String key) {
        int value = map.get(key);
        value += 1;
        map.put(key, value);
    }

    public void removeKeyIfExists(HashMap<String, Integer> map, String key) {
        if (map.get(key) != null) {
            map.remove(key);
        }
    }
}
```

Response:
```json
[
    {
        "problematic_line_of_code": "return map.get(key);",
        "description": "This may return null if the key does not exist, which may cause a NullPointerException in caller code.",
        "suggestion": "Consider checking if the key exists before returning or return a default value.",
        "action": "insert_above"
    },
    {
        "problematic_line_of_code": "int value = map.get(key);",
        "description": "If the key does not exist, map.get(key) returns null, causing a NullPointerException when unboxing to int.",
        "suggestion": "Check if the key exists and initialize value properly.",
        "action": "insert_above"
    },
    {
        "problematic_line_of_code": "if (map.get(key) != null)",
        "description": "A more efficient check for key existence is map.containsKey(key).",
        "suggestion": "Change the condition to if (map.containsKey(key)).",
        "action": "change"
    }
]
```

## Example4

Code:
```cpp
int calculate_sum_squared(std::vector<int> numbers) {
    int sum = 0;
    for (int j = 0; j < numbers.size(); j++) {
        sum -= numbers[j];
        sum *= sum;
    }
    return sum;
}

int calculate_sum(std::vector<int> numbers) {
    int sum = 0;
    for (int j = 0; j < numbers.size(); j++) {
        sum -= numbers[j];
    }
    return sum;
}

int calculate_product(std::vector<int> numbers) {
    int product = 1;
    for (int j = 0; j < numbers.size(); j++) {
        product *= numbers[i];
    }
    return product;
}
```

Response:
```json
[{  "problematic_line_of_code" : "sum -= numbers[j];",
    "description" : "The operator -= is wrong to calculate a sum. It should be +=.",
    "suggestion" : "Change the operator to +=.",
    "action": "change"},
    {"problematic_line_of_code" : "sum -= numbers[j];",
    "description" : "The operator -= is wrong to calculate a sum. It should be +=.",
    "suggestion" : "Change the operator to +=.",
    "action": "change"}]
```

# Your Task

Please review the code below and use the formatting of the examples to provide your response as a list of JSON objects.
"""

max_new_tokens = 1500
context_window_size = 8000
recursion_depth = 3
json_extraction_re = re.compile(r"```json(.*)```", re.DOTALL)


def recursive_prompting(counter, chat, prompt):
    print(f"ITERATION: {counter}\n")
    if counter > recursion_depth:
        print("Recursion depth exceeded")
        return []
    with torch.no_grad():
        chat_prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]
    if prompt_len > context_window_size - max_new_tokens:
        print("Context size exceeded")
        return []
    print(f"Prompting with context length:\n{prompt_len}<eol>")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    result = tokenizer.decode(outputs[0][prompt_len:])

    del inputs
    del outputs
    torch.cuda.empty_cache()

    print(f"OUTPUT{counter}:\n{result}\nOUTPUT_END{counter}")
    result_json = []
    matches = re.findall(json_extraction_re, result)
    if len(matches) == 0:
        print(f"JSON extraction failed:\n{result}\nJSON END")
        chat.append({"role": "assistant", "content": result})
        chat.append(
            {
                "role": "user",
                "content": """
It seems that the last output provided does not contain a valid JSON object. Please ensure that the response is a JSON array of objects, where each object contains the fields "problematic_line_of_code", "description", and "suggestion". The JSON format should look like this:

```json
[
  {
    "problematic_line_of_code": "<line of code>",
    "description": "<description of the potential bug>",
    "suggestion": "<suggestion for fixing or investigating the issue>",
    "action": "<'change', 'insert_above' or 'insert_below'>"
  },
]
```

Please try again to generate a JSON object for each line that might contain bugs in the code provided earlier, and ensure the output is formatted as valid JSON for all the items.
""",
            }
        )
        return recursive_prompting(counter + 1, chat, prompt)
    processed_result = matches[0]
    try:
        result_json = json.loads(processed_result)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed:\n{processed_result}\nJSON END")
        chat.append({"role": "assistant", "content": result})
        chat.append(
            {
                "role": "user",
                "content": f"""
It looks like the last output provided is not in the correct JSON format. Parsing the output failed with the following exception: {e} Please ensure that the response is a JSON array of objects, where each object contains the fields "problematic_line_of_code", "description", and "suggestion". The JSON format should look like this:

```json
[
  {{
    "problematic_line_of_code": "<line of code>",
    "description": "<description of the potential bug>",
    "suggestion": "<suggestion for fixing or investigating the issue>"
    "action": "<'change', 'insert_above' or 'insert_below'>"
  }},
]
```

Please try again with the code provided earlier, and ensure the output is formatted as valid JSON for all the items.
""",
            }
        )
        return recursive_prompting(counter + 1, chat, prompt)

    return_values = []
    if not isinstance(result_json, list):
        print(f"Result is not a list:\n{result_json}\n JSON END")
        result_json = [result_json]
    for finding in result_json:
        if "problematic_line_of_code" not in finding:
            print(f"Problematic line of code not found in finding {finding}")
            continue
        line_of_code = finding["problematic_line_of_code"]
        prompt = str(prompt)
        first_index = 0
        while first_index != -1:
            first_index = prompt.find(line_of_code, first_index)
            if first_index != -1:
                finding_copy = copy.deepcopy(finding)
                finding_copy["line_number"] = prompt[:first_index].count("\n") + 1
                if "description" not in finding_copy:
                    finding_copy["description"] = ""
                if "suggestion" not in finding_copy:
                    finding_copy["suggestion"] = ""
                return_values.append(finding_copy)
                first_index += 1

        if prompt.find(line_of_code) == -1:
            print(
                f"Line of code could not be found {finding['problematic_line_of_code']}"
            )
            chat.append({"role": "assistant", "content": result})
            chat.append(
                {
                    "role": "user",
                    "content": f"""
It appears that the identified substring in the "problematic_line_of_code" field in this json object does not exist in the original code provided:
```json
{json.dumps(finding)}
```
                         
Please ensure that the "problematic_line_of_code" field contains exact lines or substrings that are present in the original code.
Make sure that indentation, line breaks, and other formatting are preserved to match the original code structure.

Please analyze the code again and make sure that each identified "problematic_line_of_code" corresponds exactly to a line or substring from the original source code.
""",
                }
            )
            result = recursive_prompting(counter + 1, chat, prompt)
            counter += 1
            if len(result) > 0:
                print("Using recursion!")
                return result
            print("Discarding recursion!")

    return return_values


@app.post("/highlight-code/")
async def highlight_code(prompt: Prompt):
    content = new_prompt
    content = content + "\nCode:\n```" + prompt.language + "\n" + prompt.code + "\n```"

    if prompt.context:
        content = (
            content
            + "Here is some additional context about what the code should do:\n"
            + prompt.context
            + "\n"
        )

    if prompt.debugState:
        print(prompt.debugState)
        content = (
            content
            + "Here is JSON object that describes the dynamic program state.\n ```json\n"
            + prompt.debugState
            + "\n```\nThe current line describes where the program is halted. The stackFrames attribute is a list of the names of the stack frames. The globalVariables attribute contains information about the global variables. The localVariables attribute contains information about all local variables. Please take this information into account when identifying bugs.\n"
        )

    chat = [
        {
            "role": "user",
            "content": content,
        },
    ]
    global request_counter
    print(f"Handling prompt {request_counter} with\nCODE:{prompt.code}CODE")
    request_counter += 1
    result = recursive_prompting(0, chat, prompt.code)
    print(f"RESULT: {json.dumps(result)}RESULT_END")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
