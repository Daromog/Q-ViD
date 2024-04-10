# Question-Instructed Visual Descriptions for Zero-Shot Video Question Answering

This is the official repository for the Q-ViD paper available at: https://arxiv.org/pdf/2402.10698.pdf

## Model Description
Q-ViD performs video QA by relying on an 𝐨𝐩𝐞𝐧 instruction-aware vision-language model to generate video frame descriptions that are more relevant to the task at hand, it uses question-dependent captioning instructions as input to [InstructBLIP](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) to generate useful frame captions. Subsequently, we use a QA instruction that combines the captions, question, options and a task description as input to a LLM-based reasoning module to perform video QA. When compared with prior works based on more complex architectures or 𝐜𝐥𝐨𝐬𝐞𝐝 GPT models, Q-ViD achieves either the second-best or the best overall average accuracy across all evaluated benchmarks.

<p align="center">
  <img width="700" alt="fig1" src="https://github.com/Daromog/Q-ViD/assets/40415903/aaa95d2e-4331-45ca-8a1e-1df019a8d9a9">
  <img width="733" alt="3f" src="https://github.com/Daromog/Q-ViD/assets/40415903/429b6eec-0f1b-46ce-bfd3-2ace0a8bfac3">
</p>

Q-ViD relies on [InstructBLIP](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md), its original trimmed checkpoints that can be used by the [Lavis](https://github.com/salesforce/LAVIS/tree/main) library can be downloaded with the following links:

| Hugging Face Model  | Checkpoint |
| ------------- | ------------- |
| [InstructBLIP-flan-t5-xl](https://huggingface.co/Salesforce/instructblip-flan-t5-xl)  | [Download](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_flanxl_trimmed.pth)  |
| [InstructBLIP-flan-t5-xxl](https://huggingface.co/Salesforce/instructblip-flan-t5-xxl)  | [Download](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_flanxxl_trimmed.pth)  |

## Installation
1. (Optional) Creating conda environment
```
conda create -n Q-ViD python=3.8
conda activate Q-ViD
```
2. Build from source
```
cd Q-ViD
pip install -e .
```
## Datasets
We test Q-ViD in five video QA benchmarks. They can be downloaded in the following links: 

1. [NExT-QA](https://github.com/doc-doc/NExT-QA)
2. [STAR](https://bobbywu.com/STAR/)
3. [How2QA](https://github.com/ych133/How2R-and-How2QA?tab=readme-ov-file)
4. [TVQA](https://github.com/jayleicn/TVQA/tree/master)
5. [IntentQA](https://github.com/JoseponLee/IntentQA)

After downloading, the original files (json,jsonl,csv) from each dataset they have to be preprocessed to a unique json format using a [preprocessing_script](data_preprocessing/Data_Preprocess.ipynb).
Afterwards, update the paths for the video folder and annotation files in the corresponding [config](lavis/configs/datasets) files from each dataset.

## Usage
Update the paths depending on your directory:

### NExT-QA
```
#Path for saving results
result_dir="YOUR_PATH"

exp_name='nextqa_infer'
ckpt='Q-ViD/qvid_checkpoints/instruct_blip_flanxxl_trimmed.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 evaluate.py \
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
```

### STAR
```
# parameters/data path
result_dir="YOUR_PATH"

exp_name='star_infer'
ckpt='Q-ViD/qvid_checkpoints/instruct_blip_flanxxl_trimmed.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 evaluate.py \
--cfg-path Q-ViD/lavis/projects/qvid/star_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.model_type=flant5xxl \
model.cap_prompt="Provide a detailed description of the image related to the" \
model.qa_prompt="Considering the information presented in the captions, select the correct answer in one letter (A,B,C,D) from the options." \
datasets.star.vis_processor.eval.n_frms=64 \
run.batch_size_eval=1 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'

```

### How2QA
```
# parameters/data path
result_dir="YOUR_PATH"

exp_name='how2qa_infer'
ckpt='Q-ViD/qvid_checkpoints/instruct_blip_flanxxl_trimmed.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 evaluate.py \
--cfg-path Q-ViD/lavis/projects/qvid/how2qa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.model_type=flant5xxl \
model.cap_prompt="Provide a detailed description of the image related to the" \
model.qa_prompt="Considering the information presented in the captions, select the correct answer in one letter (A,B,C,D) from the options." \
datasets.how2qa.vis_processor.eval.n_frms=64 \
run.batch_size_eval=2 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'

```

### TVQA
```
# parameters/data path
result_dir="YOUR_PATH"

exp_name='tvqa_infer'
ckpt='Q-ViD/qvid_checkpoints/instruct_blip_flanxxl_trimmed.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 evaluate.py \
--cfg-path Q-ViD/lavis/projects/qvid/tvqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
model.model_type=flant5xxl \
model.cap_prompt="Provide a detailed description of the image related to the" \
model.qa_prompt="Considering the information presented in the captions, select the correct answer in one letter (A,B,C,D,E) from the options." \
datasets.tvqa.vis_processor.eval.n_frms=64 \
run.batch_size_eval=1 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'
```

### IntentQA
```
# parameters/data path
result_dir="YOUR_PATH"

exp_name='intentqa_infer'
ckpt='Q-ViD/qvid_checkpoints/instruct_blip_flanxxl_trimmed.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 evaluate.py \
--cfg-path Q-ViD/lavis/projects/qvid/intentqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.model_type=flant5xxl \
model.cap_prompt="Provide a detailed description of the image related to the" \
model.qa_prompt="Considering the information presented in the captions, select the correct answer in one letter (A,B,C,D,E) from the options." \
datasets.intentqa.vis_processor.eval.n_frms=64 \
run.batch_size_eval=1 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'

```
## Acknowledgments
We thank the developers of LAVIS, BLIP-2, SeViLa for their public code release.


