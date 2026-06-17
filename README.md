## vLLM Generation and Hidden-State Extraction

`LLMs/llama_vllm.py` generates model responses with vLLM, and
`LLMs/extract_coninstruct_hidden_states.py` extracts hidden states from the
ConInstruct conflict-resolution prompts.

### Generate Responses

Run the following in a Jupyter notebook:

```bash
%%bash
model=Qwen/Qwen3.5-9B

for conflict_type_idx in 1 2 3 4 5 6 7 8 9
do
  VLLM_USE_FLASHINFER_SAMPLER=0 python LLMs/llama_vllm.py \
    --model $model \
    --cache_dir "" \
    --task conflict_resolution \
    --conflict_type_idx $conflict_type_idx \
    --dtype auto \
    --trust_remote_code
done
```

### Extract Hidden States

Run the following in a Jupyter notebook:

```bash
!python LLMs/extract_coninstruct_hidden_states.py \
  --model Qwen/Qwen3.5-9B \
  --cache_dir "" \
  --torch-dtype auto \
  --trust-remote-code \
  --batch-size 16
```

# Below are from the ConInstruct repository.

## Requirements

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   
pip install bitsandbytes   
pip install accelerate transformers   
pip install nltk   

You should set your API KEY in the following files before running the scripts:  
- line 3 of LLMs/evaluation.py  
- line 5 of LLMs/gpt4.py

## ConInstruct Dataset

We begin by manually curating 100 seed instructions, which serve as fundamental instructions without additional constraints. We then use GPT-4 to incorporate six types of constraints into each seed instruction, thereby generating a set of expanded instructions. Finally, GPT-4 is employed to introduce conflicting constraints into the expanded instructions, resulting in the ConInstruct dataset.

| Instruction Type | Dataset Path |
|------|----------|
| seed instructions | `datasets/seed_instruction.jsonl` |
| expanded instructions | `datasets/expand_instruction.jsonl` |
| conflicting instructions | `datasets/conflict_instruction.jsonl` |

ConInstruct is also available at **Hugging Face**: https://huggingface.co/datasets/He-Xingwei/ConInstruct. For specific data format details, please refer to the Hugging Face link above.

## Conflict Detection

Run conflict detection on LLMs with instructions containing different types of conflicts (Results in Table 2):

```bash
sh scripts/conflict_detection.sh
```

Run conflict detection on LLMs with instructions containing different numbers of conflicts (Results in Figure 3):

```bash
sh scripts/conflict_detection_density.sh
```

## Conflict Resolution

Run the following script to get the distributions of conflict resolution behaviors exhibited by LLMs:

```bash
sh scripts/conflict_resolution_density.sh
```
