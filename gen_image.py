# ------------------------------------------
# TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering
# Paper Link: https://arxiv.org/abs/2311.16465
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser-2
# Copyright (c) Microsoft Corporation.
# ------------------------------------------

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import glob
import time

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Dataset

from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from PIL import Image
import string
alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(aphabet) = 95
'''alphabet
0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 
'''

logger = get_logger(__name__, log_level="INFO")


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="JingyeChen22/textdiffuser2-full-ft",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--stable_diffusion_model_name",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="Path to Stable Diffussion or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='lambdalabs/pokemon-blip-captions',
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=43512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="/nfs/nas-6.1/gtyi/cvpdl_final/cvpdl_project/diffusion_experiment_result_6epoch_1/checkpoint-132",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        type=bool,
        default=True
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--vis_num",
        type=int,
        default=1,
        help=("The number of images to be visualized."),
    )

    #### newly added parameters
    parser.add_argument(
        "--granularity", 
        type=int, 
        default=128, 
        help="The granularity of coordinates, ranging from 1~512."
    )
    parser.add_argument(
        "--coord_mode", 
        type=str, 
        default='ltrb',
        choices=['lt', 'center', 'ltrb'],
        help="The way to represent coordinates."
    )
    parser.add_argument(
        "--max_length", 
        default=77,
        type=int, 
        help="Maximum length of the composed prompt."
    )
    parser.add_argument(
        "--cfg", 
        default=7.5,
        type=float, 
        help="classifier free guidance."
    )
    parser.add_argument(
        "--sample_steps", 
        default=50,
        type=int, 
        help="steps for sampling for diffusion models."
    )
    parser.add_argument(
        "--input_format", 
        type=str, 
        help="specify the input format",
        default="prompt",
        choices=['prompt', 'prompts_txt_file', 'prompt_layout_txt_file']
    )
    parser.add_argument(
        "--input_prompt", 
        type=str, 
        default="a text image of hello world"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
    )
    parser.add_argument(
        "--prompts_txt_file", 
        type=str, 
    )
    parser.add_argument(
        "--m1_model_path", 
        type=str, 
        default="JingyeChen22/textdiffuser2_layout_planner",
        help="the checkpoint of layout planner"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


DATASET_NAME_MAPPING = {
    # "lambdalabs/pokemon-blip-captions": ("image", "text"),
    "MARIO-10M": ("image", "text"), 
}

def get_args(checkpoint_path):
    args = argparse.Namespace(allow_tf32=False, cfg=7.5, coord_mode='ltrb', dataset_name='lambdalabs/pokemon-blip-captions', enable_xformers_memory_efficient_attention=True, gradient_accumulation_steps=4, gradient_checkpointing=True, granularity=128, hub_model_id=None, hub_token=None, input_file=None, input_format='prompt', input_prompt='a text image of hello world', local_rank=-1, logging_dir='logs', m1_model_path='JingyeChen22/textdiffuser2_layout_planner', max_length=77, mixed_precision='no', output_dir='inference_results', pretrained_model_name_or_path='JingyeChen22/textdiffuser2-full-ft', prompts_txt_file=None, push_to_hub=False, rank=4, report_to='tensorboard', resolution=43512, resume_from_checkpoint=checkpoint_path, sample_steps=50, seed=None, stable_diffusion_model_name='stable-diffusion-v1-5/stable-diffusion-v1-5', vis_num=1)

    return args

def load_model(args=None):
    if args is None:
        args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        args.stable_diffusion_model_name, subfolder="tokenizer"
    )

    #### additional tokens are introduced, including coordinate tokens and character tokens
    print('***************')
    print(len(tokenizer))
    for i in range(520):
        tokenizer.add_tokens(['l' + str(i) ]) # left
        tokenizer.add_tokens(['t' + str(i) ]) # top
        tokenizer.add_tokens(['r' + str(i) ]) # width
        tokenizer.add_tokens(['b' + str(i) ]) # height    
    for c in alphabet:
        tokenizer.add_tokens([f'[{c}]']) 
    print(len(tokenizer))
    print('***************')

    if args.max_length == 77:
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", ignore_mismatched_sizes=True
        )
    else:
        #### enlarge the context length of text encoder. empirically, enlarging the context length can proceed longer sequence. However, we observe that it will be hard to render general objects
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", max_position_embeddings=args.max_length, ignore_mismatched_sizes=True
        )

    text_encoder.resize_token_embeddings(len(tokenizer))

    vae = AutoencoderKL.from_pretrained(args.stable_diffusion_model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype) 
    vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        )

    unet.set_attn_processor(lora_attn_procs)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = AttnProcsLayers(unet.attn_processors)

    lora_layers, text_encoder = accelerator.prepare(
        lora_layers, text_encoder
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            accelerator.load_state(args.resume_from_checkpoint)

    if accelerator.is_main_process and os.path.exists(f'{args.output_dir}'):
        print('detect existing output_dir, removing the contained jpg/txt files ...')
        os.system(f'rm {args.output_dir}/*.jpg')
        os.system(f'rm {args.output_dir}/*.txt')

    return args, (tokenizer, text_encoder, unet, vae)

# prompt = "<|startoftext|>ayurvedic balancing : an integration of western fitness with eastern wellness 9 7 8 0 7 3 8 7 0 1 8 8 2 <|endoftext|><|startoftext|> l26 t64 r101 b80 [B] [A] [L] [A] [N] [C] [I] [N] [G] bold, coral orange, artistic style <|endoftext|><|endoftext|><|startoftext|> l29 t49 r98 b61 [A] [Y] [U] [R] [V] [E] [D] [I] [C] bold, coral orange, artistic style <|endoftext|><|endoftext|><|startoftext|>"

def generate_image(args, model_objects, user_prompt):
    tokenizer, text_encoder, unet, vae = model_objects

    with torch.no_grad():

        # size = len(ocrs)
        size = 1
        print(f'the number of samples: {size}')

        time_seed = int(time.time())
        random.seed(time_seed)
        torch.manual_seed(time_seed)
        torch.cuda.manual_seed_all(time_seed)
  
        result_images = []

        for sample_index in range(size):
            prompts_cond = user_prompt

            prompts_cond = tokenizer(prompts_cond, truncation=True, return_tensors="pt").input_ids[0].tolist()

            print('prompt', prompts_cond)
            print("prompt_decoded", tokenizer.decode(prompts_cond))

            prompts_nocond = [tokenizer.pad_token_id]*args.max_length

            prompts_cond = [prompts_cond] * args.vis_num
            prompts_nocond = [prompts_nocond] * args.vis_num

            prompts_cond = torch.Tensor(prompts_cond).long().cuda()
            prompts_nocond = torch.Tensor(prompts_nocond).long().cuda()

            scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") 
            scheduler.set_timesteps(args.sample_steps) 
            noise = torch.randn((args.vis_num, 4, 64, 64)).to("cuda") 
            input = noise

            encoder_hidden_states_cond = text_encoder(prompts_cond)[0]
            encoder_hidden_states_nocond = text_encoder(prompts_nocond)[0] 

            texts = prompts_cond
            f = open(f'{args.output_dir}/prompt_{sample_index}_{args.local_rank}.txt', 'w+')
            for text in texts:
                sentence = tokenizer.decode(text)
                f.write(sentence + '\n')
            f.close()

            for t in tqdm(scheduler.timesteps):
                with torch.no_grad():  # classifier free guidance
                    noise_pred_cond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_cond[:args.vis_num]).sample # b, 4, 64, 64
                    noise_pred_uncond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_nocond[:args.vis_num]).sample # b, 4, 64, 64
                    noisy_residual = noise_pred_uncond + args.cfg * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
                    prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                    input = prev_noisy_sample

            # decode
            input = 1 / vae.config.scaling_factor * input 
            images = vae.decode(input, return_dict=False)[0] 
            width, height = 512, 512
            new_image = Image.new('RGB', (4*width, 4*height))
            for index, image in enumerate(images.float()):
                image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                row = index // 4
                col = index % 4
                new_image.paste(image, (col*width, row*height))
            new_image.save(f'{args.output_dir}/pred_img_{sample_index}_{args.local_rank}.jpg')

            result_images.append(new_image)

    return result_images[0]

if __name__ == "__main__":
    args = get_args()
    args, model_objects = load_model(args)
    image = generate_image(args, model_objects)
    print(type(image))