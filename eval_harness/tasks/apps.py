"""Measuring Coding Challenge Competence With APPS
https://arxiv.org/abs/2105.09938

APPS is a benchmark for code generation with 10000 problems. With three difficulty levels: introductory, interview and competition.
It can be used to evaluate the ability of language models to generate code from natural language specifications.

Homepage: https://github.com/hendrycks/apps
"""

import json

from evaluate import load

from eval_harness.base import Task
import re
_CITATION = """
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""


LEVELS = ["introductory", "interview", "competition"]
from math import gcd
from collections import Counter

def normalize_model_body(generation: str):#, tabsize: int):
    """
    Detect the leading indentation style of `generation` and remove ONE indent level.
    Returns (body, info) where `body` is runnable at top level and `info` describes detection.

    - Handles tabs, spaces (2/3/4/8), and mixed whitespace.
    - Doesn't require valid Python—works on raw text.
    - Preserves inner indentation (relative structure).
    """
    def detect_indentation(text: str):
        space_widths, tab_lines, mixed_ws_lines, total_indented = [], 0, 0, 0
        for ln in text.splitlines():
            if not ln.strip():
                continue
            m = re.match(r'[ \t]+', ln)
            if not m:
                continue
            total_indented += 1
            ws = m.group(0)
            kinds = set(ws)
            if kinds == {' '}:
                space_widths.append(len(ws))
            elif kinds == {'\t'}:
                tab_lines += 1
            else:
                mixed_ws_lines += 1

        if total_indented == 0:
            return {"style": "none", "unit": None, "details": "No indented lines found."}

        if tab_lines > 0 and not space_widths and mixed_ws_lines == 0:
            return {"style": "tabs", "unit": "\\t", "details": f"{tab_lines} tab-indented lines"}

        if space_widths and tab_lines == 0 and mixed_ws_lines == 0:
            uniq = sorted(set(space_widths))
            unit = uniq[0]
            for n in uniq[1:]:
                unit = gcd(unit, n)
            if unit not in (2, 3, 4, 8):
                unit = min((2, 4, 8), key=lambda k: abs(k - unit))
            return {"style": "spaces", "unit": unit, "details": f"Observed space indents: {Counter(space_widths)}"}

        return {
            "style": "mixed",
            "unit": (min(space_widths) if space_widths else None),
            "details": {
                "space_indents_seen": Counter(space_widths),
                "tab_only_lines": tab_lines,
                "lines_with_both_tabs_and_spaces": mixed_ws_lines
            }
        }

    info = detect_indentation(generation)
    body = generation.lstrip("\n")  # drop any leading blank line

    if info["style"] == "tabs":
        # Remove exactly ONE leading tab from indented lines
        body = re.sub(r'^\t', '', body, flags=re.MULTILINE)

    elif info["style"] == "spaces":
        unit = int(info["unit"])
        # Remove exactly ONE indent unit of spaces; be forgiving if a line has < unit
        pattern = re.compile(r'^(?: {' + str(unit) + r'}|[ ]{1,' + str(unit) + r'})', re.MULTILINE)
        body = pattern.sub('', body)

    return body, info

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {apps-interview: Task, apps-competitoon: Task}
    """
    return {f"apps-{level}": create_task(level) for level in LEVELS}


def create_task(level):
    class APPS(GeneralAPPS):
        def __init__(self, **kwargs):
            super().__init__(level, **kwargs)

    return APPS


class GeneralAPPS(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "codeparrot/apps"
    DATASET_NAME = None

    def __init__(self, level, k_list=[1, 5, 10, 25, 100]):
        self.DATASET_NAME = level
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```", "\ndef"],
            requires_execution=True,
        )
        self.k_list = k_list

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Generate prompts for APPS
        Finetuning setup: prompt=question  with some starter code and function name if they exist.
        We also specify the type of the prompt, i.e. whether it is call-based or standard input-based.
        """
        starter_code = None if len(doc["starter_code"]) == 0 else doc["starter_code"]
        if starter_code is not None:
            print("starter_code", starter_code)
        try:
            input_outpout = json.loads(doc["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None
      #  print("fn_name", fn_name)
      #  prompt = "\nQUESTION:\n"
        prompt = doc["question"]
        if starter_code:
            prompt += starter_code
        if not fn_name:
            #call_format = "\nUse Standard Input format"
            prompt += "\n\ndef solve():\n" #call_format
        #else:
        #    call_format = "\nUse Call-Based format"
        #    prompt += "def solve():" #call_format
        #prompt += "\nANSWER:\n"
        return prompt
    

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return None

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for APPS)
        """
     #   try:
     #       generation = generation.split("\nANSWER:", 1)[1]
     #   except IndexError:
     #       # happens when prompts were very long and got truncated
     #       pass
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]
        generation = self._stop_at_stop_token(generation, self.stop_words)
        print("preegeneration", generation)
        generation, _ = normalize_model_body(generation)
        print("aftergeneration", generation)
        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences (not needed for APPS Task)
        """
        code_metric = load("codeparrot/apps_metric")
       # if level is None:
        level = self.DATASET_NAME
        results = code_metric.compute(
            predictions=generations, k_list=self.k_list, level=self.DATASET_NAME
        )
        return results
