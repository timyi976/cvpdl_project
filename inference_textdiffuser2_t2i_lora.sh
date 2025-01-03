accelerate launch inference_t2i_only.py \
  --pretrained_model_name_or_path="JingyeChen22/textdiffuser2-full-ft" \
  --stable_diffusion_model_name="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="no" \
  --output_dir="inference_results" \
  --enable_xformers_memory_efficient_attention \
  --resume_from_checkpoint="/nfs/nas-6.1/gtyi/cvpdl_final/cvpdl_project/diffusion_experiment_result_6epoch_1/checkpoint-132" \
  --granularity=128 \
  --coord_mode="ltrb" \
  --cfg=7.5 \
  --sample_steps=50 \
  --seed=43555 \
  --m1_model_path="JingyeChen22/textdiffuser2_layout_planner" \
  --input_format='prompt' \
  --input_prompt='a stamp of u.s.a'