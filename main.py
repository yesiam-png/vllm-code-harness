import os
import json
import datasets
import torch
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM
from eval_harness.arguments import (
    GenerationArguments,
    ModelArguments,
    VLLMArguments,
    WorkflowArguments,
    pattern_match
)
from eval_harness.evaluator import Evaluator
from eval_harness.tasks import ALL_TASKS
import copy
import gc

def _parse_models_arg(model_arg: str):
    """Allow comma-separated models; otherwise single."""
    if "," in model_arg:
        return [m.strip() for m in model_arg.split(",") if m.strip()]
    return [model_arg.strip()]

def _model_label(path: str) -> str:
    """Short, readable label for table rows."""
    return os.path.basename(os.path.normpath(path)) or path

def _model_slug(path: str) -> str:
    """Safe slug for filenames."""
    return _model_label(path).replace("/", "_").replace(" ", "_")

def _make_metric_path_for_model(base_path: str, model_slug: str, task_name: str, run_name: str) -> str:
    """Attach model, task, run to metric filename to avoid overwrites."""
    root = base_path[:-5] if base_path.endswith(".json") else base_path
    return f"{root}.{model_slug}.{task_name}.{run_name}.json"

def _fmt_pct(x):
    if x is None:
        return "-"
    try:
        return f"{100.0*float(x):.1f}%"
    except Exception:
        return str(x)

def _print_summary_table(summary, tasks_order):
    """
    summary: dict[model_label][task] -> dict with keys 'p1' and 'p10', each a dict possibly containing 'pass@1'/'pass@10'
    tasks_order: list of task names in desired column order
    """
    headers = ["model"]
    for t in tasks_order:
        headers += [f"{t} p@1", f"{t} p@10"]

    rows = []
    for model_label, per_task in summary.items():
        row = [model_label]
        for t in tasks_order:
            r_p1 = (per_task.get(t, {}).get("p10", {}) or {}).get("pass@1")
            if r_p1 is None:
                r_p1 = (per_task.get(t, {}).get("p1", {}) or {}).get("pass@1")
            r_p10 = (per_task.get(t, {}).get("p10", {}) or {}).get("pass@10")
            row += [_fmt_pct(r_p1), _fmt_pct(r_p10)]
        rows.append(row)

    # simple width calc and print
    cols = [headers] + rows
    widths = [max(len(str(r[i])) for r in cols) for i in range(len(headers))]

    def _print_row(r):
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))

    _print_row(headers)
    print("-+-".join("-"*w for w in widths))
    for r in rows:
        _print_row(r)


def main():
    parser = HfArgumentParser([GenerationArguments, ModelArguments, VLLMArguments, WorkflowArguments])
    args = parser.parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    # --- multi-task + two-pass + multi-model evaluation ---

    # tasks: allow comma-separated
    tasks_list = pattern_match([t.strip() for t in args.tasks.split(",") if t.strip()], ALL_TASKS)

    # models: allow comma-separated in --model
    models_list = _parse_models_arg(args.model)

    # define the two evaluation passes
    eval_runs = [
        ("p1",  {"n_samples": 1,                           "temperature": 0.0}),
        ("p10", {"n_samples": max(10, args.n_samples),     "temperature": args.temperature}),
    ]

    # helper to build per-run metric path including model
    def make_metric_path(base_path: str, model_path: str, task_name: str, run_name: str) -> str:
        return _make_metric_path_for_model(base_path, _model_slug(model_path), task_name, run_name)

    all_results = {}          # nested: model -> task -> run_name -> metrics
    table_summary = {}        # nested: model_label -> task -> {'p1': {...}, 'p10': {...}}

    if args.load_generations_path:
        print("evaluation only mode")
        for model_path in models_list:
            model_label = _model_label(model_path)
            all_results[model_label] = {}
            table_summary[model_label] = {}

            for task in tasks_list:
                print(f"[TASK] {task}")
                all_results[model_label][task] = {}
                table_summary[model_label][task] = {}

                for run_name, conf in eval_runs:
                    print(f"[pass@] {run_name}")
                    run_args = copy.deepcopy(args)
                    run_args.n_samples = conf["n_samples"]
                    run_args.temperature = conf["temperature"]

                    evaluator = Evaluator(None, None, run_args)
                    out = evaluator.evaluate(task)
                    all_results[model_label][task][run_name] = out
                    table_summary[model_label][task][run_name] = out

                    # write per-model/per-task/per-run metrics
                    metric_path = make_metric_path(run_args.metric_output_path, model_path, task, run_name)
                    metric_dir = os.path.dirname(metric_path)
                    if metric_dir:
                        os.makedirs(metric_dir, mode=0o755, exist_ok=True)
                    payload = {task: {run_name: out},
                            "config": {**vars(run_args), "task": task, "run": run_name, "model": model_path}}
                    with open(metric_path, "w+") as f:
                        f.write(json.dumps(payload, indent=2))

    else:
        # shared tokenizer options (rebuilt per model)
        dict_precisions = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if args.precision not in dict_precisions:
            raise ValueError(f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16")

        for model_path in models_list:
            model_label = _model_label(model_path)
            print(f"\n=== Loading model: {model_path} ===")
            # build tokenizer
            if args.left_padding:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=args.trust_remote_code,
                    use_auth_token=args.use_auth_token,
                    padding_side="left",
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=args.trust_remote_code,
                    use_auth_token=args.use_auth_token,
                    truncation_side="left",
                    padding_side="right",
                )
            if not tokenizer.eos_token:
                if tokenizer.bos_token:
                    tokenizer.eos_token = tokenizer.bos_token
                    print("bos_token used as eos_token")
                else:
                    raise ValueError("No eos_token or bos_token found")
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except AttributeError:
                print("Not setting pad_token to eos_token")
                pass

            WIZARD_LLAMA_MODELS = [
                "WizardLM/WizardCoder-Python-34B-V1.0",
                "WizardLM/WizardCoder-34B-V1.0",
                "WizardLM/WizardCoder-Python-13B-V1.0",
            ]
            if model_path in WIZARD_LLAMA_MODELS:
                tokenizer.bos_token = "<s>"
                tokenizer.bos_token_id = 1
                print("Changing bos_token to <s>")

            # build model
            model = LLM(
                model=model_path,
                tensor_parallel_size=1,
                dtype=dict_precisions[args.precision],
                trust_remote_code=args.trust_remote_code,
                gpu_memory_utilization=args.gpu_memory_utilization,
                swap_space=args.swap_space,
                max_seq_len_to_capture=args.sequence_length_limit,
                max_model_len=args.sequence_length_limit,
            )
            model.set_tokenizer(tokenizer=tokenizer)

            # evaluate all tasks and both passes
            all_results[model_label] = {}
            table_summary[model_label] = {}

            for task in tasks_list:
                all_results[model_label][task] = {}
                table_summary[model_label][task] = {}
                for run_name, conf in eval_runs:
                    run_args = copy.deepcopy(args)
                    run_args.n_samples = conf["n_samples"]
                    run_args.temperature = conf["temperature"]
                    # IMPORTANT: ensure downstream code sees the current model path
                    run_args.model = model_path

                    evaluator = Evaluator(model, tokenizer, run_args)

                    if run_args.generation_only:
                        print(f"generation mode only [{model_label} | {task} | {run_name}]")
                        generations, references = evaluator.generate_text(task)
                        evaluator.save_json_files(
                            generations,
                            references,
                            run_args.save_generations_path,
                            run_args.save_references_path,
                        )
                        out = {"saved_generations_to": run_args.save_generations_path,
                            "saved_references_to": run_args.save_references_path}
                    else:
                        print(f"evaluating [{model_label} | {task} | {run_name}] ...")
                        out = evaluator.evaluate(task)

                    all_results[model_label][task][run_name] = out
                    table_summary[model_label][task][run_name] = out

                    metric_path = make_metric_path(run_args.metric_output_path, model_path, task, run_name)
                    metric_dir = os.path.dirname(metric_path)
                    if metric_dir:
                        os.makedirs(metric_dir, mode=0o755, exist_ok=True)
                    payload = {task: {run_name: out},
                            "config": {**vars(run_args), "task": task, "run": run_name, "model": model_path}}
                    with open(metric_path, "w+") as f:
                        f.write(json.dumps(payload, indent=2))

            # free VRAM before next model
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # --- end multi-model block ---



    # Save all args to config and print summary table
    final_results = {"results": all_results, "config": vars(args)}
    dumped = json.dumps(final_results, indent=2)
    print(dumped)

    metric_dir = os.path.dirname(args.metric_output_path)
    if metric_dir:
        os.makedirs(metric_dir, mode=0o755, exist_ok=True)
    with open(args.metric_output_path, "w+") as f:
        f.write(dumped)

    # Print a compact cross-model table
    print("\n=== Summary (rows=models; columns=task pass@1 / pass@10) ===")
    _print_summary_table(table_summary, tasks_list)



if __name__ == "__main__":
    main()
