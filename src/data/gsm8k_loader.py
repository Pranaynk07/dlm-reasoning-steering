"""GSM8K loader with CoT vs Direct prompt formatting for contrastive analysis."""

import re
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

COT_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step, showing your work.\n\n"
    "Question: {question}\n\n"
    "Solution: Let me solve this step by step.\n"
)

DIRECT_PROMPT_TEMPLATE = (
    "Answer the following math question with just the final number.\n\n"
    "Question: {question}\n\n"
    "Answer: The answer is"
)

COT_PROMPT_VARIANTS = [
    "Solve step by step:\n{question}\n\nStep 1:",
    "Let's work through this math problem carefully.\n\nProblem: {question}\n\nFirst,",
    "Show your reasoning for the following problem.\n\n{question}\n\nReasoning:",
]

DIRECT_PROMPT_VARIANTS = [
    "What is the answer? {question}\nAnswer:",
    "{question}\nThe final answer is",
    "Compute: {question}\nResult:",
]


class GSM8KLoader:
    def __init__(self, split="test", n_problems=200, cache_dir=None, seed=42):
        self.split = split
        self.n_problems = n_problems
        self.cache_dir = cache_dir
        self.seed = seed
        self.problems = []
        self._load()

    def _load(self):
        logger.info(f"Loading GSM8K {self.split}...")
        dataset = load_dataset("openai/gsm8k", "main", split=self.split, cache_dir=self.cache_dir)
        dataset = dataset.shuffle(seed=self.seed)
        if self.n_problems < len(dataset):
            dataset = dataset.select(range(self.n_problems))

        self.problems = [
            {
                'question': item['question'],
                'answer': item['answer'],
                'numerical_answer': self._extract_answer(item['answer']),
                'idx': i,
            }
            for i, item in enumerate(dataset)
        ]
        logger.info(f"Loaded {len(self.problems)} problems")

    @staticmethod
    def _extract_answer(answer_text):
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                return None
        return None

    def _format_prompts(self, template, prompt_type):
        return [
            {
                'prompt': template.format(question=p['question']),
                'question': p['question'], 'answer': p['numerical_answer'],
                'full_answer': p['answer'], 'idx': p['idx'], 'prompt_type': prompt_type,
            }
            for p in self.problems
        ]

    def get_cot_prompts(self, variant=0):
        tmpl = COT_PROMPT_TEMPLATE if variant == 0 else COT_PROMPT_VARIANTS[min(variant-1, len(COT_PROMPT_VARIANTS)-1)]
        return self._format_prompts(tmpl, 'cot')

    def get_direct_prompts(self, variant=0):
        tmpl = DIRECT_PROMPT_TEMPLATE if variant == 0 else DIRECT_PROMPT_VARIANTS[min(variant-1, len(DIRECT_PROMPT_VARIANTS)-1)]
        return self._format_prompts(tmpl, 'direct')

    def get_contrastive_pairs(self, variant=0):
        return list(zip(self.get_cot_prompts(variant), self.get_direct_prompts(variant)))

    def split_discovery_eval(self, discovery_frac=0.5):
        n = int(len(self.problems) * discovery_frac)
        return GSM8KSubset(self.problems[:n], 'discovery'), GSM8KSubset(self.problems[n:], 'eval')


class GSM8KSubset:
    def __init__(self, problems, name="subset"):
        self.problems = problems
        self.name = name

    def __len__(self):
        return len(self.problems)

    def get_cot_prompts(self, variant=0):
        tmpl = COT_PROMPT_TEMPLATE if variant == 0 else COT_PROMPT_VARIANTS[min(variant-1, len(COT_PROMPT_VARIANTS)-1)]
        return [
            {'prompt': tmpl.format(question=p['question']), 'question': p['question'],
             'answer': p['numerical_answer'], 'idx': p['idx'], 'prompt_type': 'cot'}
            for p in self.problems
        ]

    def get_direct_prompts(self, variant=0):
        tmpl = DIRECT_PROMPT_TEMPLATE if variant == 0 else DIRECT_PROMPT_VARIANTS[min(variant-1, len(DIRECT_PROMPT_VARIANTS)-1)]
        return [
            {'prompt': tmpl.format(question=p['question']), 'question': p['question'],
             'answer': p['numerical_answer'], 'idx': p['idx'], 'prompt_type': 'direct'}
            for p in self.problems
        ]


def parse_generated_answer(text):
    for pattern in [r'####\s*(-?[\d,]+\.?\d*)', r'(?:answer\s+is|=)\s*\$?(-?[\d,]+\.?\d*)']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                continue
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    return None
