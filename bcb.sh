# split: full
OUTPUT_DIR="/mnt/task_runtime/vllm-code-harness/log_bcb"
mkdir -p ${OUTPUT_DIR}/bigcodebench/full
python benchmarks/bigcodebench/generate.py \
    --model "/mnt/task_runtime/40-400-qwen-10warmup-5penalty-log-005lenpenalty-3sync/global_step_2400" \
    --split complete \
    --subset full \
    --greedy  \
    --bs 1 \
    --temperature 0 \
    --n_samples 1 \
    --resume  \
    --backend vllm \
    --tp 1 \
    --save_path ${OUTPUT_DIR}/bigcodebench/full/completion.jsonl 

python benchmarks/bigcodebench/sanitize.py \
    --samples ${OUTPUT_DIR}/bigcodebench/full/completion.jsonl \
    --calibrate

python benchmarks/bigcodebench/evaluate.py \
    --split complete \
    --subset full \
    --no-gt \
    --samples ${OUTPUT_DIR}/bigcodebench/full/completion-sanitized-calibrated.jsonl