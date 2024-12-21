# CVPDL Project

***Text Styling Based on Textdiffuser-2***

## Files

- Modified from the original repo
    ```
    # Training script
    train_textdiffuser2_t2i_lora.py
    train_textdiffuser2_t2i_lora.sh
    # Inference script
    inference_textdiffuser2_t2i_lora.py
    inference_textdiffuser2_t2i_lora.sh
    ```

- Our own scripts
    ```
    # Dataset related
    download_images.py
    generate_dataset.py
    # Inference and Demo
    inference_t2i_only.py
    gen_layout.py
    gen_style_prompt.py
    gen_image.py
    frontend.py
    ```

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

## Dataset

- Run following command.
    ```bash
    python3 generate_dataset.py
    ```

## Training

- Edit argument `--output_dir` in `train_textdiffuser2_t2i_lora.sh` accordingly.

- Run the script, and the LoRA checkpoint will be saved in the designated directory.
    ```bash
    bash train_textdiffuser2_t2i_lora.sh
    ```

## Demo and Inference

- Create a file to store your OpenAI API key.

- Run following command, then open the provided Gradio web URL in your browser.
    ```bash
    python3 frontend.py --api_key_file [your OpenAI API key file] --lora_ckpt [path to your trained LoRA checkpoint]
    ```

## Reference

- Original Textdiffuser-2: [https://github.com/microsoft/unilm/tree/master/textdiffuser-2](https://github.com/microsoft/unilm/tree/master/textdiffuser-2)
- Hugging Face Diffuser Example scripts: [https://github.com/huggingface/diffusers/tree/main/examples/text_to_image](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)