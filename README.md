# CVPDL Project

## Env Install

- Install Pytorch
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

- Install packages
    ```bash
    pip3 install -r requirements.txt
    ```

- Install xformers
    ```bash
    pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
    ```

## TODO

- Generate text color and style description label for some of the images in the dataset
- Edit `preprocess_train()` in `train_textdiffuser2_t2i_lora.py` to put text style description in the image prompt
- `train_textdiffuser2_t2i_lora.py` finetune from the original Stable Diffusion v1.5 model, need to modifiy the code in order to finetune on their Text Diffuser 2 model
- Use Lora to finetune model
- Write inference script
- Create Gradio interface for demo

## Resources

- Original repo: [https://github.com/microsoft/unilm/tree/master/textdiffuser-2](https://github.com/microsoft/unilm/tree/master/textdiffuser-2)
- Hugging Face Diffuser Example scripts: [https://github.com/huggingface/diffusers/tree/main/examples/text_to_image](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)