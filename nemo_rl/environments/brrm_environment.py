# Two-Stage Unified Quality Assessment GenRM Implementation
import re
import json
import logging
from typing import Any, Optional, TypedDict, Tuple, Dict
import numpy as np
import ray
import torch
from dataclasses import dataclass

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

# ========================= STAGE 1: UNIFIED QUALITY ASSESSMENT =========================

UNIFIED_ANALYSIS_PROMPT = """You are a response quality evaluator. Given the context of the conversation (the last turn is the User's query) and two responses from the Assistant, you should compare the difference of two model responses, select the most important cognitive abilities for this query, and analyze critical issues in each response.

**Context:** 
{context}

**Responses:**
{responses}

**Output Format:**
[Quality Assessment Focus]
Choose 1-3 abilities: Information Accuracy, Computational Precision, Logical Reasoning, Implementation Capability, Safety Awareness, Response Completeness, Instruction Adherence, Communication Clarity.
[End of Quality Assessment Focus]

[Quality Analysis for Response 1]
- Critical Issues: [Focus on chosen abilities, list specific errors/concerns, or "None identified"]
  * Information Accuracy: factual errors, source reliability, misinformation
  * Computational Precision: calculation errors, formula mistakes, step validity
  * Logical Reasoning: conclusion correctness (CRITICAL), logical flaws, reasoning gaps
  * Implementation Capability: functional errors (CRITICAL), security issues, inefficiency
  * Safety Awareness: harmful content (CRITICAL), inappropriate refusals, bias
  * Instruction Adherence: constraint violations, format errors, requirement misses
  * Response Completeness: missing content, insufficient detail, incomplete coverage
[End of Quality Analysis for Response 1]

[Quality Analysis for Response 2]
- Critical Issues: [Same format as above]
[End of Quality Analysis for Response 2]"""


# ========================= STAGE 2: SCORING =========================


SCORING_STAGE_PROMPT = """You are making final comparative judgments using established evaluation priorities. You have the conversation context, two responses to compare, and a detailed quality analysis from a previous evaluation. 
Before scoring, analyze step by step. Different query types require different evaluation hierarchies. Please strictly follow the required output format.

**Evaluation Hierarchies:**
- **Accuracy-Critical** (factual, computational, technical): Correctness > Process > Presentation 
- **Creative/Open-Ended** (writing, discussion): User Intent > Content Quality > Creativity 
- **Instruction-Following** (constrained tasks): Adherence > Content > Clarity

#### Output Format Requirements ####
[The Begin of Analysis on Response 1]
[Apply appropriate evaluation hierarchy to the quality analysis findings]
[The End of Analysis on Response 1]

[The Begin of Analysis on Response 2]
[Apply appropriate evaluation hierarchy to the quality analysis findings]
[The End of Analysis on Response 2]

[The Begin of Ranking Score]
\\boxed{1 or 2} (response that better meets the appropriate evaluation hierarchy)
[The End of Ranking Score]
"""

# ========================= DATA STRUCTURES =========================

class TwoStageMetadata(TypedDict):
    num_responses: int
    preference: Optional[int]
    quality_assessment_complete: Optional[bool]  # was: factcheck_stage_complete
    quality_assessment_results: Optional[str]    # was: factcheck_results
    context: Optional[str]
    response1: Optional[str]
    response2: Optional[str]

# ========================= PROMPT FORMATTING =========================

def format_unified_analysis_prompt(context: str, response1: str, response2: Optional[str] = None) -> str:
    """Format the GenRM prompt with context and responses."""
    if response2 is None:
        responses = f"[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n"
    else:
        responses = f"[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n\n[The Begin of Response 2]\n{response2}\n[The End of Response 2]\n"

    return UNIFIED_ANALYSIS_PROMPT.format(
        context=f"[The Begin of Context]\n{context}\n[The End of Context]\n",
        responses=responses
    )

def format_scoring_stage_prompt(context: str, response1: str, response2: Optional[str], quality_results: str) -> str:
    """Format the scoring stage prompt with quality analysis results."""
    return SCORING_STAGE_PROMPT

# ========================= PARSING UTILITIES =========================

def parse_unified_analysis_response(response: str, num_responses: int = 2) -> tuple[bool, str]:
    """
    Parse unified quality analysis response.
    Returns (is_valid, formatted_analysis)
    """
    try:
        # Extract quality assessment focus
        focus_match = re.search(r'\[Quality Assessment Focus\]\s*(.*?)\s*\[End of Quality Assessment Focus\]', 
                               response, flags=re.DOTALL | re.IGNORECASE)
        
        if focus_match:
            focus_content = focus_match.group(1).strip()
        else:
            focus_content = "Primary Evaluation Dimensions: General Communication\nKey Concerns: Overall response quality"
        
        # Extract analysis blocks
        blocks = {}
        for idx in range(num_responses):
            response_id = idx + 1
            pattern = rf"\[Quality Analysis for Response {response_id}\]\s*(.*?)\s*\[End of Quality Analysis for Response {response_id}\]"
            match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                blocks[response_id] = content
        
        # Reconstruct the analysis maintaining the quality assessment narrative
        structured_parts = [f"[Quality Assessment Focus]\n{focus_content}\n[End of Quality Assessment Focus]"]
        
        for response_id in blocks:
            content = blocks[response_id]
            structured_block = f"[Quality Analysis for Response {response_id}]\n{content}\n[End of Quality Analysis for Response {response_id}]"
            structured_parts.append(structured_block)
        
        return True, "\n\n".join(structured_parts)

    except Exception as e:
        return False, "No valid quality analysis results."

def parse_scoring_response(assistant_response: str, num_responses: int) -> tuple[bool, Optional[list], Optional[str], str]:
    """
    Validate the strict format and extract scores.
    Returns: (is_valid_format, preference, error_message)
    """
    if num_responses == 2:
        # Check ranking score section
        ranking_pattern = r'\[The Begin of Ranking Score\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Ranking Score\]'
        ranking_match = re.search(ranking_pattern, assistant_response, re.DOTALL)
        
        if not ranking_match:
            return False, None, "Format violation: Missing or incorrect '[The Begin of Ranking Score]\\n\\\\boxed{z} \\n[The End of Ranking Score]' section"
        
        # Extract preference ranking
        preference_content = ranking_match.group(1).strip()
        # Preference ranking should be a single value
        if ',' in preference_content:
            return False, None, "Format violation: Preference ranking should be a single value, found comma"
        
        return True, preference_content, ""
    
    else:
        return False, None, f"Unsupported number of responses: {num_responses}"


# ========================= UTILITY FUNCTIONS =========================

def parse_score_value(score_str: str) -> Optional[int]:
    """Parse a score value from a string, handling various formats."""
    if not score_str:
        return None
        
    score_str = score_str.strip()

    try:
        return int(score_str)
    except ValueError:
        pass
    
    match = re.search(r'(\d+)', score_str)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass     
    return None
    



def distance_abs(predicted: Any, ground_truth: Any, penalty=100) -> int:
    value = -1 * penalty
    """Calculate absolute distance between predicted and ground truth."""
    try:
        # Handle None values
        if predicted is None or ground_truth is None:
            return value
        
        # Convert to integers if they're strings
        if isinstance(predicted, str):
            predicted = parse_score_value(predicted)
            if predicted is None:
                return value
                
        if isinstance(ground_truth, str):
            ground_truth = parse_score_value(ground_truth)
            if ground_truth is None:
                return value
        
        # Calculate distance
        return abs(int(predicted) - int(ground_truth)) * 10

    except Exception as e:
        logging.error(f"Error calculating distance: {e}, predicted: {predicted}, ground_truth: {ground_truth}")
        return value





# ========================= TWO-STAGE ENVIRONMENT =========================

@ray.remote
class BRRMEnvironment(EnvironmentInterface):
    """Two-stage unified quality assessment environment: Stage 1 quality analysis, Stage 2 score."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.format_penalty = cfg.get("format_penalty", -100)
        self.quality_bonus_multiplier = cfg.get("quality_bonus_multiplier", 0.0)
        logging.basicConfig(level=logging.INFO)
    

    def step(self, message_log_batch: list[list[dict[str, str]]], metadata: list[TwoStageMetadata]) -> EnvironmentReturn:
        """Process two-stage quality assessment and scoring."""
        
        print(f"\n[TWO-STAGE ENV] Processing batch of {len(message_log_batch)} samples")
        
        rewards = []
        observations = []
        next_metadata = []
        answers = []

        for i, (conversation, meta) in enumerate(zip(message_log_batch, metadata)):
            # Extract assistant's response
            rm_response = ""
            for msg in reversed(conversation):
                if msg["role"] == "assistant":
                    rm_response = msg["content"]
                    break
            
            # Check which stage we're in
            if not meta.get("quality_assessment_complete"):
                # STAGE 1: Quality assessment
                reward, obs, updated_meta, answer = self._process_quality_assessment_stage(
                    rm_response, meta
                )
                # CRITICAL: Ensure we move to next stage
                if updated_meta and not updated_meta.get("quality_assessment_complete"):
                    print(f"[WARNING] Sample {i} didn't complete quality assessment stage properly")
            else:
                # STAGE 2: Scoring
                reward, obs, updated_meta, answer = self._process_scoring_stage(
                    rm_response, meta
                )
            
            rewards.append(reward)
            observations.append(obs)
            next_metadata.append(updated_meta)
            answers.append(answer)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # Only terminate after scoring stage
        terminateds = torch.tensor([
            meta is None for meta in next_metadata
        ], dtype=torch.bool)

        # Add stop strings for each stage
        next_stop_strings = []
        for meta in next_metadata:
            if meta and not meta.get("quality_assessment_complete"):
                # Quality assessment stage - stop after quality analysis output
                next_stop_strings.append(["[End of Quality Analysis for Response 2]"])
            else:
                # Scoring stage - stop after ranking score
                next_stop_strings.append(["[The End of Ranking Score]"])
        
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=next_stop_strings,  # Use stage-specific stop strings
            rewards=rewards_tensor,
            terminateds=terminateds,
            answers=answers,
        )

    def _process_quality_assessment_stage(self, response: str, metadata: TwoStageMetadata) -> Tuple[float, dict, TwoStageMetadata]:
        """Process quality assessment stage."""
        
        # Parse and store quality analysis results
        is_parsed, parsed_response = parse_unified_analysis_response(response)
        
        # Store quality analysis results and prepare for scoring stage
        updated_metadata = metadata.copy()
        updated_metadata["quality_assessment_complete"] = True
        updated_metadata["quality_assessment_results"] = parsed_response
        
        # IMPORTANT: Create the observation that will be used as the next prompt
        # This should be the scoring stage prompt
        scoring_prompt = format_scoring_stage_prompt(
            metadata["context"],
            metadata["response1"], 
            metadata["response2"],
            parsed_response
        )
        
        reward = 0.0 if is_parsed else float(self.format_penalty)

        # Return observation that becomes the next user message
        obs = {"role": "user", "content": "<|im_start|>user\n" + scoring_prompt + "<|im_end|>\n<|im_start|>assistant\n"}
        answer = None
        return reward, obs, updated_metadata, answer


    def _process_scoring_stage(self, response: str, metadata: TwoStageMetadata) -> Tuple[float, dict, TwoStageMetadata]:
        """Process scoring stage with extensive debugging."""
        
        # Parse scoring response (still need this for reward calculation)
        is_valid, preference, error_msg = parse_scoring_response(
            response, metadata["num_responses"]
        )

        if error_msg:
            print(f"  Parse error: {error_msg}")
        
        if not is_valid:
            print(f"[SCORING STAGE] Parse error: {error_msg}")
            obs = {
                "role": "environment",
                "content": f"<environment>Scoring stage format error: {error_msg}</environment>"
            }
            return float(self.format_penalty), obs, None, None
        
        # Calculate base reward from scoring accuracy
        reward = self.format_penalty
        try:
            if metadata["num_responses"] == 2:
                # Calculate distance for preference ranking
                if metadata["preference"] is not None and preference:
                    reward = - distance_abs(preference, metadata["preference"], self.format_penalty)
                
        except Exception as e:
            print(f"[SCORING STAGE] Score parsing error: {e}")
            obs = {
                "role": "environment",
                "content": f"<environment>Score parsing error: {e}</environment>"
            }
            return float(self.format_penalty), obs, None, None

        obs = {
            "role": "environment",
            "content": f"<environment>Two-stage completed. Final reward: {reward}</environment>",
        }
        return float(reward), obs, None, preference  # Terminate episode

    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Calculate metrics for unified quality assessment approach."""
        rewards = batch.get("rewards", torch.tensor([]))
        num_samples = len(batch.get("idx", []))
        
        if len(rewards) == 0:
            return batch, {}
        
        rewards_np = rewards.numpy() if hasattr(rewards, 'numpy') else np.array(rewards)
        
        # Calculate metrics
        mean_reward = float(np.mean(rewards_np))
        format_violation_rate = float(np.mean(rewards_np == self.format_penalty))
        
        # For valid rewards, calculate performance metrics
        valid_rewards = rewards_np[rewards_np != self.format_penalty]
        if len(valid_rewards) > 0:
            mean_valid_reward = float(np.mean(valid_rewards))
            positive_reward_rate = float(np.mean(valid_rewards > 0))
            high_reward_rate = float(np.mean(valid_rewards > -5))  # Less than 5 points penalty
        else:
            mean_valid_reward = 0.0
            positive_reward_rate = 0.0
            high_reward_rate = 0.0
        
        metrics = {
            "mean_reward": mean_reward,
            "format_violation_rate": format_violation_rate,
            "mean_valid_reward": mean_valid_reward,
            "positive_reward_rate": positive_reward_rate,
            "high_reward_rate": high_reward_rate,
            "num_samples": num_samples,
            "valid_samples": len(valid_rewards),
            "approach": "unified_quality_assessment",  # Updated approach name
        }
        
        return batch, metrics
    
    def shutdown(self):
        """Clean up resources."""
        pass