result_dir=""

exp_name='nextqa_infer'
ckpt='Q-ViD/qvid_checkpoints/instruct_blip_flanxxl_trimmed.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 Q-ViD/evaluate.py \
--cfg-path Q-ViD/lavis/projects/qvid/nextqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.model_type=flant5xxl \
model.cap_prompt="Provide a detailed description of the image related to the" \
model.qa_prompt="Considering the information presented in the captions, select the correct answer in one letter (A,B,C,D,E) from the options." \
datasets.nextqa.vis_processor.eval.n_frms=64 \
run.batch_size_eval=2 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'
