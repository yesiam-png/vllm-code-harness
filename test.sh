#!/usr/bin/env bash
set -euo pipefail
#rm -rf runs
#rm -rf logs
#rm -rf bcb_results
#rm -rf results
# === Edit these to your setup ===
TASKS="mbpp,humaneval-unstripped"
MAXLEN=512

IS_ROOT="${IS_ROOT:-1}"

GPUS=(0 1 2 3 4 5 6 7)
STEPS=(200 400 600 800 1000 1200 1600 2000)  # or STEPS=({200..2400..200})

#MODEL_ROOT="${MODEL_ROOT:-}"  # allow env override: MODEL_ROOT=... ./run.sh
#MODEL_ROOT="${MODEL_ROOT%\\}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# Accept: ./run.sh model_root=...   or   ./run.sh --model-root=...   or   ./run.sh --model-root ... 
while (($#)); do
  case "$1" in
    model=*|MODEL_ROOT=*)
      MODEL_ROOT="${1#*=}"; shift ;;
    --model=*)
      MODEL_ROOT="${1#*=}"; shift ;;
    --model)
      shift
      MODEL_ROOT="${1:-}"; shift || true ;;
    *)
      # keep other args intact if you add more flags later
      shift ;;
  esac
done

MODEL_ROOT="${MODEL_ROOT%/}"

# Require it
if [[ -z "${MODEL_ROOT}" ]]; then
  echo "ERROR: MODEL_ROOT not set.
Usage examples:
  ./run.sh model=../openandsyn-qwen-ntponly
  ./run.sh --model ../openandsyn-qwen-ntponly
  MODEL_ROOT=../openandsyn-qwen-ntponly ./run.sh" >&2
  exit 1
fi

# Derive RUN_TAG after MODEL_ROOT is known (still allow override via env)
RUN_TAG="${RUN_TAG:-$(basename "$(realpath "$MODEL_ROOT")")}"

OUT_DIR="runs/${RUN_TAG}"
LOG_DIR="logs/${RUN_TAG}"
BCB_DIR="${BCB_DIR:-bcb_results}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

num_gpus=${#GPUS[@]}
total=${#STEPS[@]}
offset=0

log "Run tag: ${RUN_TAG}"
log "Model root: ${MODEL_ROOT}"
log "Tasks: ${TASKS}"
log "Steps: ${STEPS[*]}"
log "GPUs: ${GPUS[*]}"
echo

while [[ $offset -lt $total ]]; do
  for (( j=0; j<num_gpus && offset+j<total; j++ )); do
    step=${STEPS[$((offset+j))]}
    gpu=${GPUS[$j]}
    base_step_path="${MODEL_ROOT}/global_step_${step}"

    # Real model dir we need to load
    model_real="$base_step_path"
    if [[ "$IS_ROOT" != "1" ]]; then
      model_real="${model_real}/actor/huggingface"
    fi

    # Create a per-step alias whose basename is global_step_${step}
    # so the harness slug becomes "global_step_${step}" instead of "huggingface".
    alias_root="${OUT_DIR}/_model_aliases"
    mkdir -p "$alias_root"
    model_alias="${alias_root}/global_step_${step}"
    ln -sfn "$(realpath "$model_real")" "$model_alias"

    # Use the alias for evaluation so filenames include the step
    model_for_cli="$model_alias"

    out_base="${OUT_DIR}/step_${step}"
    mkdir -p "$out_base"

    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      log "[GPU ${gpu}] START step ${step}  tasks=${TASKS}  model=${model_for_cli} -> $(realpath "$model_real")"
      python main.py \
        --model "$model_for_cli" \
        --max_length_generation "$MAXLEN" \
        --tasks "$TASKS" \
        --temperature 0.6 \
        --n_samples 10 \
        --allow_code_execution \
        --metric_output_path "${out_base}/metrics.json" \
        > "${LOG_DIR}/step_${step}.log" 2>&1
      rc=$?
      if [[ $rc -eq 0 ]]; then
        log "[GPU ${gpu}] DONE  step ${step}"
      else
        log "[GPU ${gpu}] FAIL  step ${step} (exit ${rc}) — see ${LOG_DIR}/step_${step}.log"
      fi
      exit $rc
    ) &
  done
  wait
  offset=$((offset+num_gpus))
done



# Echo where to find artifacts (helps distinguish runs)
echo $MODEL_ROOT
echo "All jobs done."

# --- Status snapshot for THIS run (which task/pass files exist) ---
OUT_DIR_PY="$OUT_DIR" TASKS_PY="$TASKS" python - <<'PY'
import os, glob, re
from collections import defaultdict

OUT_DIR = os.environ.get("OUT_DIR_PY")
TASKS = os.environ.get("TASKS_PY","")
if not OUT_DIR:
    raise SystemExit("OUT_DIR_PY env missing")

task_list = [t.strip() for t in TASKS.split(",") if t.strip()]
steps = sorted({
    int(p.split("_")[-1])
    for p in glob.glob(os.path.join(OUT_DIR, "step_*"))
    if p.split("_")[-1].isdigit()
})

# Presence map: pres[step][task] = {"p1": bool, "p10": bool}
pres = defaultdict(lambda: defaultdict(lambda: {"p1": False, "p10": False}))
for step in steps:
    base = os.path.join(OUT_DIR, f"step_{step}")
    for f in glob.glob(os.path.join(base, "metrics.*.*.p*.json")):
        m = re.match(r".*metrics\.(.+?)\.(.+?)\.(p1|p10)\.json$", f)
        if not m: 
            continue
        _, task, run = m.groups()
        pres[step][task][run] = True

headers = ["step"]
for t in task_list:
    headers += [f"{t} p@1", f"{t} p@10"]

rows = []
for step in steps:
    row = [str(step)]
    for t in task_list:
        state = pres[step][t]
        row += [("OK" if state["p1"] else "--"), ("OK" if state["p10"] else "--")]
    rows.append(row)

widths = [len(h) for h in headers]
for r in rows:
    for i,c in enumerate(r):
        widths[i] = max(widths[i], len(c))

def pr(r): print(" | ".join(str(c).ljust(w) for c,w in zip(r,widths)))
print("\n=== File presence (this run only) ===")
pr(headers)
print("-+-".join("-"*w for w in widths))
for r in rows: pr(r)
PY

# --- Aggregated summary table for THIS run ---
OUT_DIR_PY="$OUT_DIR" python - <<'PY'
import os, json, glob, re
from collections import defaultdict

OUT_DIR = os.environ["OUT_DIR_PY"]

files = glob.glob(os.path.join(OUT_DIR, "step_*", "metrics.*.*.p*.json"))
table = defaultdict(lambda: defaultdict(lambda: {"p1": None, "p10": None}))

def pct(x): return "-" if x is None else f"{100*float(x):.1f}%"

for f in files:
    base = os.path.basename(f)  # metrics.<model_slug>.<task>.<p1|p10>.json
    m = re.match(r"metrics\.(.+?)\.(.+?)\.(p1|p10)\.json$", base)
    if not m: 
        continue
    model_slug, task, run = m.groups()
    with open(f) as fp:
        data = json.load(fp)
    out = data.get(task, {}).get(run, {})
    if run == "p1" and "pass@1" in out:
        table[model_slug][task]["p1"] = out["pass@1"]
    if run == "p10" and "pass@10" in out:
        table[model_slug][task]["p10"] = out["pass@10"]

tasks = sorted({t for per_model in table.values() for t in per_model.keys()})
headers = ["model"] + [h for t in tasks for h in (f"{t} p@1", f"{t} p@10")]
widths = [len(h) for h in headers]

rows = []
for model, per_task in sorted(table.items()):
    row = [model]
    for t in tasks:
        row += [pct(per_task[t]["p1"]), pct(per_task[t]["p10"])]
    rows.append(row)
    for i,c in enumerate(row):
        widths[i] = max(widths[i], len(str(c)))

def pr(r): print(" | ".join(str(c).ljust(w) for c,w in zip(r,widths)))
print("\n=== Summary (this run only) ===")
pr(headers)
print("-+-".join("-"*w for w in widths))
for r in rows: pr(r)
PY


SKIP_BCB="${SKIP_BCB:-0}"   # set SKIP_BCB=1 to skip BigCodeBench and show only MBPP/HumanEval tables
# =========================
# 2) BigCodeBench runs (optional)
# =========================
if [[ "$SKIP_BCB" != "1" ]]; then
  pip install -q --upgrade bigcodebench
  pip install --upgrade protobuf
  pip install numpy==1.26.4
  pip install tensorflow==2.20.0

  # BCB Hard (Complete split)
  offset=0
  while [[ $offset -lt $total ]]; do
    for (( j=0; j<num_gpus && offset+j<total; j++ )); do
      step=${STEPS[$((offset+j))]}
      gpu=${GPUS[$j]}
      model="${MODEL_ROOT}/global_step_${step}"
      if [[ "$IS_ROOT" != "1" ]]; then
        model="${model}/actor/huggingface"
      fi
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        log "[GPU ${gpu}] BCB HARD START step ${step} model=${model}"
        bigcodebench.evaluate \
          --model "$model" \
          --split complete \
          --subset hard \
          --backend vllm \
          --parallel 64 \
          --resume false \
          --direct_completion \
          --gradio_endpoint https://zhangshenao-bigcodebench-evaluator.hf.space \
          --pass_k 1 --n_samples 1 --temperature 0.0 \
          > "${LOG_DIR}/bcb_hard_step_${step}.log" 2>&1 || true
        log "[GPU ${gpu}] BCB HARD DONE  step ${step}"
      ) &
    done
    wait
    offset=$((offset+num_gpus))
  done

  # BCB Full (Complete split)
  offset=0
  while [[ $offset -lt $total ]]; do
    for (( j=0; j<num_gpus && offset+j<total; j++ )); do
      step=${STEPS[$((offset+j))]}
      gpu=${GPUS[$j]}
      model="${MODEL_ROOT}/global_step_${step}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        log "[GPU ${gpu}] BCB FULL START step ${step} model=${model}"
        bigcodebench.evaluate \
          --model "$model" \
          --split complete \
          --subset full \
          --backend vllm \
          --parallel 64 \
          --resume false \
          --direct_completion \
          --gradio_endpoint https://zhangshenao-bigcodebench-evaluator.hf.space \
          --pass_k 1 --n_samples 1 --temperature 0.0 \
          > "${LOG_DIR}/bcb_full_step_${step}.log" 2>&1 || true
        log "[GPU ${gpu}] BCB FULL DONE  step ${step}"
      ) &
    done
    wait
    offset=$((offset+num_gpus))
  done

  echo
  log "All jobs finished."
  echo "$MODEL_ROOT"
  echo
fi


# =========================
# 5) Extended Summary (+ BigCodeBench Hard/Full)  (only if not skipped)
# =========================
if [[ "$SKIP_BCB" != "1" ]]; then
  OUT_DIR_PY="$OUT_DIR" BCB_DIR_PY="$BCB_DIR" MODEL_ROOT_PY="$(realpath "$MODEL_ROOT")" python - <<'PY'
import os, json, glob, re
from collections import defaultdict

OUT_DIR = os.environ["OUT_DIR_PY"]
BCB_DIR = os.environ.get("BCB_DIR_PY", "bcb_results")
MODEL_ROOT = os.environ.get("MODEL_ROOT_PY","")

# Build rows from harness metrics (same as previous table)
files = glob.glob(os.path.join(OUT_DIR, "step_*", "metrics.*.*.p*.json"))
rows_by_model = defaultdict(lambda: defaultdict(lambda: {"p1": None, "p10": None}))
step_to_model = {}  # map step -> model_slug in our rows

def pct(x): return "-" if x is None else f"{100*float(x):.1f}%"

for f in files:
    step_dir = os.path.basename(os.path.dirname(f))  # step_400
    mstep = re.match(r"step_(\d+)$", step_dir)
    step = int(mstep.group(1)) if mstep else None

    base = os.path.basename(f)  # metrics.<model_slug>.<task>.<p1|p10>.json
    m = re.match(r"metrics\.(.+?)\.(.+?)\.(p1|p10)\.json$", base)
    if not m: 
        continue
    model_slug, task, run = m.groups()
    step_to_model[step] = model_slug

    with open(f) as fp:
        data = json.load(fp)
    out = data.get(task, {}).get(run, {})
    if run == "p1" and "pass@1" in out:
        rows_by_model[model_slug][task]["p1"] = out["pass@1"]
    if run == "p10" and "pass@10" in out:
        rows_by_model[model_slug][task]["p10"] = out["pass@10"]

# Parse BigCodeBench pass@k outputs, restricted to MODEL_ROOT
bcb = defaultdict(lambda: {"hard": None, "full": None})
for f in glob.glob(os.path.join(BCB_DIR, "*pass_at_k.json")):
    try:
        with open(f) as fp:
            d = json.load(fp)
    except Exception:
        continue
    if str(d.get("split","")).lower() != "complete":
        continue
    subset = str(d.get("subset","")).lower()
    if subset not in ("hard","full"):
        continue
    model_name = str(d.get("model",""))
    # filter: only include results whose model path contains our MODEL_ROOT (avoid cross-run leakage)
    if MODEL_ROOT and MODEL_ROOT not in model_name:
        continue
    m = re.search(r"global_step_(\d+)", model_name)
    if not m:
        continue
    step = int(m.group(1))
    v = (d.get("pass@1") or
         d.get("calibrated-pass@1") or
         d.get("calibrated_pass@1") or
         (d.get("pass_at_k", {}).get("1") if isinstance(d.get("pass_at_k"), dict) else None))
    if v is None:
        continue
    bcb[step][subset] = float(v)

# Build extended header order
tasks = sorted({t for per_model in rows_by_model.values() for t in per_model.keys()})
headers = ["model"] + [h for t in tasks for h in (f"{t} p@1", f"{t} p@10")] + ["BigCodeBench Hard p@1", "BigCodeBench Full p@1"]
widths = [len(h) for h in headers]

# Compose rows with BCB columns
rows = []
for step, model in sorted(step_to_model.items()):
    row = [model]
    for t in tasks:
        row += [pct(rows_by_model[model][t]["p1"]), pct(rows_by_model[model][t]["p10"])]
    row += [pct(bcb[step]["hard"]), pct(bcb[step]["full"])]
    rows.append(row)
    for i,c in enumerate(row):
        widths[i] = max(widths[i], len(str(c)))

def pr(r): print(" | ".join(str(c).ljust(w) for c,w in zip(r,widths)))
print("\n=== Summary + BigCodeBench (this run only) ===")
pr(headers)
print("-+-".join("-"*w for w in widths))
for r in rows: pr(r)
PY
fi
