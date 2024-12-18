#### prepare m1 (layout planner)
from fastchat.model import load_model, get_conversation_template
import torch

def load_layout_planner_model(model_path="JingyeChen22/textdiffuser2_layout_planner"):
    m1_model, m1_tokenizer = load_model(
        model_path,
        'cuda',
        1,
        None,
        False,
        False,
        revision="main",
        debug=False,
    )

    return m1_model, m1_tokenizer

def generate(m1_model, m1_tokenizer, prompts):
    ocrs = []
    user_prompts = []
    for prompt in prompts:
        user_prompt = prompt
        user_prompts.append(user_prompt)
        template = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. All keywords are included in the caption. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {user_prompt}'
        msg = template
        conv = get_conversation_template("JingyeChen22/textdiffuser2_layout_planner")
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = m1_tokenizer([prompt], return_token_type_ids=False)
        inputs = {k: torch.tensor(v).to('cuda') for k, v in inputs.items()}
        output_ids = m1_model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.0,
            max_new_tokens=512,
        )

        if m1_model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = m1_tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        # print(f"[{conv.roles[0]}]\n{msg}")
        # print(f"[{conv.roles[1]}]\n{outputs}")
        # ocrs = outputs.split('\n')
        ocrs.append(outputs.strip().split('\n'))

    return ocrs

if __name__ == "__main__":
    m1_model, m1_tokenizer = load_layout_planner_model()
    prompts = ['a text image of hello world']
    ocrs = generate(m1_model, m1_tokenizer, prompts)

    print(ocrs)