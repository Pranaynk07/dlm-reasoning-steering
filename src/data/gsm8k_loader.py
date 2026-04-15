"""
GSM8K Data Loader with Chain-of-Thought vs Direct Answer prompts.

Creates paired datasets for contrastive analysis:
- CoT prompts: Encourage step-by-step mathematical reasoning
- Direct prompts: Encourage direct numerical answers

This pairing is essential for identifying reasoning-associated SAE features.
"""

import re
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


# Prompt templates for contrastive analysis
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

# Alternative prompt variants for robustness
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
    """
    Loads GSM8K dataset with contrastive prompt formatting.
    
    Key design: Each problem is formatted TWO ways —
    1. Chain-of-thought (CoT) prompt → encourages reasoning
    2. Direct answer prompt → encourages direct response
    
    The differential activation between these conditions identifies
    features that encode "reasoning behavior" in the DLM.
    """
    
    def __init__(
        self,
        split: str = "test",
        n_problems: int = 200,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Args:
            split: Dataset split ("train" or "test")
            n_problems: Number of problems to load
            cache_dir: HuggingFace cache directory
            seed: Random seed for subset selection
        """
        self.split = split
        self.n_problems = n_problems
        self.cache_dir = cache_dir
        self.seed = seed
        
        self.dataset = None
        self.problems = []
        
        self._load()
    
    def _load(self):
        """Load GSM8K from HuggingFace."""
        logger.info(f"Loading GSM8K {self.split} split...")
        
        dataset = load_dataset(
            "openai/gsm8k", "main",
            split=self.split,
            cache_dir=self.cache_dir,
        )
        
        # Shuffle and take subset
        dataset = dataset.shuffle(seed=self.seed)
        
        if self.n_problems < len(dataset):
            dataset = dataset.select(range(self.n_problems))
        
        self.dataset = dataset
        self.problems = [
            {
                'question': item['question'],
                'answer': item['answer'],
                'numerical_answer': self._extract_answer(item['answer']),
                'idx': i,
            }
            for i, item in enumerate(dataset)
        ]
        
        logger.info(f"Loaded {len(self.problems)} GSM8K problems")
    
    @staticmethod
    def _extract_answer(answer_text: str) -> Optional[float]:
        """
        Extract the numerical answer from GSM8K answer text.
        
        GSM8K answers end with "#### <number>"
        """
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
        if match:
            # Remove commas from numbers like "1,234"
            num_str = match.group(1).replace(',', '')
            try:
                return float(num_str)
            except ValueError:
                return None
        return None
    
    def get_cot_prompts(self, variant: int = 0) -> List[Dict]:
        """
        Get all problems formatted with chain-of-thought prompts.
        
        Args:
            variant: Which prompt template variant to use (0 = default)
        
        Returns:
            List of dicts with 'prompt', 'question', 'answer', 'idx'
        """
        if variant == 0:
            template = COT_PROMPT_TEMPLATE
        else:
            template = COT_PROMPT_VARIANTS[min(variant - 1, len(COT_PROMPT_VARIANTS) - 1)]
        
        return [
            {
                'prompt': template.format(question=p['question']),
                'question': p['question'],
                'answer': p['numerical_answer'],
                'full_answer': p['answer'],
                'idx': p['idx'],
                'prompt_type': 'cot',
            }
            for p in self.problems
        ]
    
    def get_direct_prompts(self, variant: int = 0) -> List[Dict]:
        """
        Get all problems formatted with direct answer prompts.
        
        Args:
            variant: Which prompt template variant to use (0 = default)
        
        Returns:
            List of dicts with 'prompt', 'question', 'answer', 'idx'
        """
        if variant == 0:
            template = DIRECT_PROMPT_TEMPLATE
        else:
            template = DIRECT_PROMPT_VARIANTS[min(variant - 1, len(DIRECT_PROMPT_VARIANTS) - 1)]
        
        return [
            {
                'prompt': template.format(question=p['question']),
                'question': p['question'],
                'answer': p['numerical_answer'],
                'full_answer': p['answer'],
                'idx': p['idx'],
                'prompt_type': 'direct',
            }
            for p in self.problems
        ]
    
    def get_contrastive_pairs(self, variant: int = 0) -> List[Tuple[Dict, Dict]]:
        """
        Get paired (CoT, Direct) prompts for each problem.
        
        Returns:
            List of (cot_prompt, direct_prompt) tuples
        """
        cot = self.get_cot_prompts(variant)
        direct = self.get_direct_prompts(variant)
        return list(zip(cot, direct))
    
    def split_discovery_eval(
        self, 
        discovery_frac: float = 0.5,
    ) -> Tuple['GSM8KSubset', 'GSM8KSubset']:
        """
        Split problems into discovery (feature identification) and 
        evaluation (steering assessment) sets.
        
        Args:
            discovery_frac: Fraction for discovery set
        
        Returns:
            (discovery_set, eval_set) as GSM8KSubset objects
        """
        n_discovery = int(len(self.problems) * discovery_frac)
        
        discovery_problems = self.problems[:n_discovery]
        eval_problems = self.problems[n_discovery:]
        
        return (
            GSM8KSubset(discovery_problems, 'discovery'),
            GSM8KSubset(eval_problems, 'eval'),
        )


class GSM8KSubset:
    """A subset of GSM8K problems with prompt formatting."""
    
    def __init__(self, problems: List[Dict], name: str = "subset"):
        self.problems = problems
        self.name = name
    
    def __len__(self):
        return len(self.problems)
    
    def get_cot_prompts(self, variant: int = 0) -> List[Dict]:
        template = COT_PROMPT_TEMPLATE if variant == 0 else COT_PROMPT_VARIANTS[min(variant - 1, len(COT_PROMPT_VARIANTS) - 1)]
        return [
            {
                'prompt': template.format(question=p['question']),
                'question': p['question'],
                'answer': p['numerical_answer'],
                'idx': p['idx'],
                'prompt_type': 'cot',
            }
            for p in self.problems
        ]
    
    def get_direct_prompts(self, variant: int = 0) -> List[Dict]:
        template = DIRECT_PROMPT_TEMPLATE if variant == 0 else DIRECT_PROMPT_VARIANTS[min(variant - 1, len(DIRECT_PROMPT_VARIANTS) - 1)]
        return [
            {
                'prompt': template.format(question=p['question']),
                'question': p['question'],
                'answer': p['numerical_answer'],
                'idx': p['idx'],
                'prompt_type': 'direct',
            }
            for p in self.problems
        ]


def parse_generated_answer(text: str) -> Optional[float]:
    """
    Extract a numerical answer from model-generated text.
    
    Handles various formats:
    - "The answer is 42"
    - "#### 42"
    - "= 42"
    - Just "42" at the end
    """
    # Try "#### number" format first
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    # Try "answer is/= number"
    match = re.search(r'(?:answer\s+is|=)\s*\$?(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    # Try last number in text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    
    return None
