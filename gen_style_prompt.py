from openai import OpenAI
import json

if True:
    openai_api_key = "TYPE_YOUR_API_KEY_HERE"

def set_api_key(key):
    global openai_api_key
    openai_api_key = key

def prompt_template(user_prompt, keywords):
    task_prompt = """You will be given a prompt by an user that will be used to generate an image, and several keywords that need to be added into generated image. Your task involves two steps.
1. You need to remove the text styling related words from the original user prompt.
2. For each keyword provided, assign a style information according to the user prompt if provided, otherwise, add a suitable style. The style description should be concise and should not exceed 6 words.

Here are two examples.

### Example 1

User Prompt: A book cover that says "Hello World" in orange, bold, modern font.Keywords: "Hello", "World"
New prompt: A book cover that says "Hello World".
Keyword with style: ```json
{"Hello": "orange, bold, modern", "World": "orange, bold, modern"}
```

### Example 2

User Prompt: A cap that has blue and bold "Teacher" and "Chair" on it.
New prompt: A cap that has "Teacher" and "Chair" on it.
Keyword with style: ```json
{"Teacher": "blue, bold", "Chair": "blue, bold"}
```

You only need to put your answer in the following format:
```json
{"prompt": 'A book cover that says "Hello World"', "styles": {"Hello": "orange, bold, modern", "World": "orange, bold, modern"}}
```"""
    template = f"Below is the user prompt and the keywords for you.\n\nUser Prompt: {user_prompt}\nKeywords: {', ' .join(keywords)}"

    ret = task_prompt + "\n\n" + template

    return ret

# print(prompt_template('A cap that has blue and bold "Teacher" and "Chair" on it.', ["Teacher", "Chair"]))
# prompt_template('A bottle has green, sleek and bold font of "Energy" and "Drink" on it.', ["Energy", "Drink"])

def generate(prompt):
    client = OpenAI(api_key=openai_api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # print(completion.choices[0].message)

    return completion.choices[0].message.content

def split_keyword_layout(ocrs):
    keywords = []
    layouts = []

    for ocr in ocrs:
        keyword, layout = ocr.split(" ")
        layout = [int(x) for x in layout.split(",")]
        keywords.append(keyword)
        layouts.append(layout)

    return keywords, layouts

def formulate_keyword(keyword, layout, style):
    prefix = "lrtb"
    ret = "<|startoftext|>"

    for p, co in zip(prefix, layout):
        ret += f" {p}{co}"

    for c in keyword:
        ret += f" [{c}]"

    ret += f" {style}"
    ret += " <|endoftext|><|endoftext|>"

    return ret

def generate_style(user_prompt, ocrs):
    keywords, layouts = split_keyword_layout(ocrs)

    raw_response = generate(prompt_template(user_prompt, keywords))
    # try to parse the response
    # remove ```json and ``` from the response
    response = raw_response.replace("```json", "").replace("```", "").strip()
    # parse tmp with json
    response = json.loads(response)

    ret = f"<|startoftext|> {response['prompt']} <|endoftext|>"
    for keyword, layout in zip(keywords, layouts):
        ret += formulate_keyword(keyword, layout, response['styles'][keyword])

    ret += "<|endoftext|><|endoftext|>"

    return ret

if __name__ == "__main__":
    # response = generate(prompt_template('A bottle has green, sleek and bold font of "Energy" and "Drink" on it.', ["Energy", "Drink"]))
    # print(response)
    response = generate_style('A bottle has green, sleek and bold font of "Energy" and "Drink" on it.', ["Energy 3,3,59,25", "Drink 1,24,58,47"])
    print(response)