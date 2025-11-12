import argparse
import os
import json
import random
import re
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple




def calculate_accuracy_by_sample_and_domain(data: list[dict]):
    """
    Calculate accuracy grouped by sample_id and also by domain.
    An example (group with same sample_id) is correct only if ALL predictions are correct.
    
    Returns:
        tuple: (overall_accuracy, domain_accuracies, sample_count, domain_sample_counts)
    """
    # Group data by sample_id
    samples_by_id = defaultdict(list)
    
    for record in data:
        metadata = record.get('metadata')
        sample_id = metadata.get('sample_id')
        domain = metadata.get('domain')
        group_id = domain + "_" + str(sample_id)
        samples_by_id[group_id].append(record)
    
    if not samples_by_id:
        return 0.0, {}, 0, {}
    
    # Calculate accuracy for each sample and track by domain
    domain_results = defaultdict(list)  # domain -> list of sample results (0 or 1)
    sample_results = []
    
    for group_id, records in samples_by_id.items():
        # Check if all predictions in this sample are correct
        all_correct = True
        sample_domain = None
        
        for record in records:
            predicted_ranking = record.get('predicted_ranking')
            metadata = record.get('metadata')
            preference = metadata.get('preference')
            
            # Try to find domain in different possible locations
            sample_domain = metadata.get('domain') 

            if predicted_ranking != preference:
                all_correct = False
                # Don't break here - we want to check for domain consistency
        
        # Record result for this sample
        sample_result = 1.0 if all_correct else 0.0
        sample_results.append(sample_result)
        
        if sample_domain is not None:
            domain_results[sample_domain].append(sample_result)
    
    # Calculate overall accuracy
    overall_accuracy = sum(sample_results) / len(sample_results) if sample_results else 0.0
    
    # Calculate accuracy for each domain
    domain_accuracies = {}
    domain_sample_counts = {}
    for domain, results in domain_results.items():
        domain_accuracies[domain] = sum(results) / len(results) if results else 0.0
        domain_sample_counts[domain] = len(results)
    
    return overall_accuracy, domain_accuracies, len(sample_results), domain_sample_counts

def calculate_step_accuracies(directory_path: str, dataset: str) -> dict:
    """
    Calculates the accuracy for all step_*_results.json files in a directory.
    Groups by sample_id and calculates accuracy per domain.
    """
    step_results = {}

    # Regex to match filenames and extract the step number
    file_pattern = re.compile(rf'step_(\d+)_{dataset}_results\.json')
    print(f"Scanning directory: '{directory_path}'...")

    try:
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            match = file_pattern.match(filename)
            # If the filename matches our pattern
            if match:
                step_number = int(match.group(1))
                file_path = os.path.join(directory_path, filename)
                
                print(f"  Processing file: {filename} (Step: {step_number})")

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_data = json.load(f)

                    if not isinstance(original_data, list):
                        print(f"    [WARNING] Content of {filename} is not a list, skipping.")
                        continue
                    
                    data = original_data

                    if data:
                        overall_accuracy, domain_accuracies, total_samples, domain_sample_counts = calculate_accuracy_by_sample_and_domain(data)
                        if dataset in ["rmb", "rewardbench"]:
                            overall_accuracy = np.mean(list(domain_accuracies.values()))
                        step_results[step_number] = {
                            'overall_accuracy': overall_accuracy,
                            'domain_accuracies': domain_accuracies,
                            'total_samples': total_samples,
                            'domain_sample_counts': domain_sample_counts,
                            'total_data_points': len(data)
                        }
                        
                        print(f"    Processed {len(data)} data points grouped into {total_samples} samples")
                        
                    else:
                        step_results[step_number] = {
                            'overall_accuracy': 0.0,
                            'domain_accuracies': {},
                            'total_samples': 0,
                            'domain_sample_counts': {},
                            'total_data_points': 0
                        }
                    
                except json.JSONDecodeError:
                    print(f"    [ERROR] File {filename} is not a valid JSON, skipping.")
                except Exception as e:
                    print(f"    [ERROR] An unknown error occurred while processing {filename}: {e}")

    except FileNotFoundError:
        print(f"[ERROR] Directory not found: '{directory_path}'")
        return {}

    # Sort the results by step number for clear presentation
    sorted_results = dict(sorted(step_results.items()))
    
    return sorted_results



def print_eval_scores(dataset, final_results):

    if final_results:

        print("\n" + "="*80)
        print(f"{dataset.upper()} EVALUATION RESULTS")
        print("="*80)
        
        for step, results in final_results.items():
            overall_acc = results['overall_accuracy']
            domain_accs = results['domain_accuracies']
            total_samples = results['total_samples']
            domain_counts = results['domain_sample_counts']
            total_data_points = results['total_data_points']
            
            print(f"\nStep {step}:")
            print(f"  Overall Accuracy: {overall_acc:.2%} ({total_samples} samples, {total_data_points} data points)")
            
            if domain_accs:
                print(f"  Domain Accuracies:")
                for domain, acc in sorted(domain_accs.items()):
                    sample_count = domain_counts.get(domain)
                    print(f"    {domain}: {acc:.2%} ({sample_count} samples)")
            else:
                print(f"  No domain information found")
                
    else:
        print("No matching files or data were found in the specified directory.")






def extract_rmbench_preference(result: dict) -> Tuple[int, str]:
    """
    Extract preference from GenRM output using the EXACT same logic as original script.
    Returns (mapped_prediction, method_used) where mapped_prediction is 0 or 1
    """
    try:
        predicted_ranking = result["predicted_ranking"]
        chosen_is_better = 1 if predicted_ranking == 1 else 0
        return chosen_is_better
        
    except Exception as e:
        chosen_is_better = random.choice([0, 1])
        return chosen_is_better
    


def group_rmbench_results_by_sample(results: List[dict]) -> Dict[str, List[dict]]:
    """Group evaluation results by original sample ID."""
    grouped = defaultdict(list)
    
    for result in results:
        metadata = result.get("metadata", {})
        sample_id = metadata.get("sample_id", f"unknown_{result.get('idx', 0)}")
        grouped[sample_id].append(result)
    
    return dict(grouped)


def compute_rmbench_accuracy_for_sample(sample_results: List[dict]) -> Dict[str, Any]:
    """Compute RM-Bench accuracy for a single sample (3x3 matrix)."""
    # Initialize 3x3 matrix for chosen vs rejected comparisons
    # Rows: chosen response styles (0=concise, 1=detailed_plain, 2=detailed_markdown)
    # Cols: rejected response styles (0=concise, 1=detailed_plain, 2=detailed_markdown)
    comparison_matrix = np.zeros((3, 3))
    comparison_counts = np.zeros((3, 3))
    

    
    for result in sample_results:
        metadata = result.get("metadata", {})
        
        # Extract metadata
        domain = metadata.get("domain")
        sample_id = metadata.get("sample_id")
        chosen_style_idx = metadata.get("chosen_style_idx")
        rejected_style_idx = metadata.get("rejected_style_idx")
        gt = metadata.get("preference")
        
        # Extract preference using the robust method (same as original script)
        chosen_is_better = extract_rmbench_preference(result)
        is_chosen_first = (gt == 0)
        comparison_counts[chosen_style_idx, rejected_style_idx] += 1
        if chosen_is_better != -1:
            if not is_chosen_first:
                # If chosen was second in the prompt, flip the preference
                chosen_is_better = not chosen_is_better
            if chosen_is_better:
                comparison_matrix[chosen_style_idx, rejected_style_idx] += 1
    
    # Normalize by counts to get accuracy matrix
    acc_matrix = np.divide(comparison_matrix, comparison_counts, 
                          out=np.zeros_like(comparison_matrix), 
                          where=comparison_counts!=0)
    
    # Compute hard, normal, easy accuracy according to RM-Bench definition
    MATRIX_SIZE = 3
    
    # Hard accuracy: upper-right triangle (chosen less fancy vs rejected more fancy)
    upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count if upper_right_count > 0 else 0.0
    
    # Normal accuracy: diagonal (same styles)
    normal_acc = np.mean(np.diag(acc_matrix))
    
    # Easy accuracy: lower-left triangle (chosen more fancy vs rejected less fancy)
    lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count if lower_left_count > 0 else 0.0
    
    # Total average accuracy
    total_avg_acc = np.mean(acc_matrix)
    
    # Merge safety subcategories into single safety domain
    if domain and domain.startswith("safety"):
        domain = "safety"
    
    return {
        "sample_id": sample_id,
        "domain": domain,
        "hard_acc": hard_acc,
        "normal_acc": normal_acc,
        "easy_acc": easy_acc,
        "total_avg_acc": total_avg_acc,
        "acc_matrix": acc_matrix.tolist(),
        "comparison_counts": comparison_counts.tolist(),
    }


def print_rmbench_results(metrics: Dict[str, Any]):
    """Print RM-Bench results in a formatted way."""
    if not metrics:
        print("No metrics to display")
        return
    
    print("\n" + "="*80)
    print("RM-BENCH EVALUATION RESULTS")
    print("="*80)
    
    # Sort by step number
    sorted_steps = sorted(metrics.keys())
    
    for step in sorted_steps:
        step_data = metrics[step]
        print(f"\nStep {step}:")
        #print("-" * 40)
        
        # Print overall metrics
        
        overall = step_data.get("overall", {})
        if overall:
            print(f"Overall Metrics (samples: {overall.get('sample_count', 0)}):")
            print(f"  Hard Accuracy:      {overall.get('hard_acc', 0):.3f}")
            print(f"  Normal Accuracy:    {overall.get('normal_acc', 0):.3f}")
            print(f"  Easy Accuracy:      {overall.get('easy_acc', 0):.3f}")
            print(f"  Total Avg Accuracy: {overall.get('total_avg_acc', 0):.3f}")
        

        # Print domain-specific metrics
        
        domains = step_data.get("domains", {})
        try:
            print(f"\nDomain-specific Metrics:")
            for domain, domain_data in sorted(domains.items()):
                print(f"  {domain} (samples: {domain_data.get('sample_count', 0)}):")
                #print(f"    Hard Acc:   {domain_data.get('hard_acc', 0):.3f}")
                #print(f"    Normal Acc: {domain_data.get('normal_acc', 0):.3f}")
                #print(f"    Easy Acc:   {domain_data.get('easy_acc', 0):.3f}")
                print(f"    Total Avg:  {domain_data.get('total_avg_acc', 0):.3f}")

        except Exception as e:
            print("error when printing results")
        
        
        


def compute_rmbench_metrics(directory_path: str, dataset: str = "rmbench") -> Dict[str, Any]:
    file_pattern = re.compile(rf'step_(\d+)_{dataset}_results\.json')
    
    all_metrics = {}
    
    try:
        for filename in os.listdir(directory_path):
            match = file_pattern.match(filename)
            if match:
                step_number = int(match.group(1))
                file_path = os.path.join(directory_path, filename)
                
                print(f"Processing step {step_number}: {filename}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    if not isinstance(results, list):
                        print(f"Warning: {filename} does not contain a list of results")
                        continue
                    
                    # Group results by sample ID
                    grouped_results = group_rmbench_results_by_sample(results)
                    
                    # Compute metrics for each sample
                    sample_metrics = []
                    all_extraction_methods = defaultdict(int)
                    
                    for sample_id, sample_results in grouped_results.items():
                        if len(sample_results) > 0:  # Should have 9 results per sample (3x3 matrix)
                            sample_metric = compute_rmbench_accuracy_for_sample(sample_results)
                            sample_metrics.append(sample_metric)
                            
                            # Aggregate extraction method statistics
                            for method, count in sample_metric.get("extraction_methods", {}).items():
                                all_extraction_methods[method] += count
                    
                    if not sample_metrics:
                        print(f"Warning: No valid samples found in {filename}")
                        continue
                    
                    # Aggregate metrics by domain
                    domain_metrics = defaultdict(list)
                    for metric in sample_metrics:
                        domain = metric["domain"]
                        domain_metrics[domain].append(metric)
                    
                    # Calculate averages for each domain
                    step_metrics = {"step": step_number, "domains": {}, "overall": {}}
                    
                    all_samples = []
                    for domain, domain_samples in domain_metrics.items():
                        if domain_samples:
                            domain_hard_acc = np.mean([s["hard_acc"] for s in domain_samples])
                            domain_normal_acc = np.mean([s["normal_acc"] for s in domain_samples])
                            domain_easy_acc = np.mean([s["easy_acc"] for s in domain_samples])
                            domain_total_avg_acc = np.mean([s["total_avg_acc"] for s in domain_samples])
                            
                            step_metrics["domains"][domain] = {
                                "hard_acc": domain_hard_acc,
                                "normal_acc": domain_normal_acc,
                                "easy_acc": domain_easy_acc,
                                "total_avg_acc": domain_total_avg_acc,
                                "sample_count": len(domain_samples)
                            }
                            
                            all_samples.extend(domain_samples)
                    
                    # Calculate overall metrics by averaging domain metrics (like Code 1)
                    if step_metrics["domains"]:
                        overall_hard_acc = np.mean([domain_data["hard_acc"] for domain_data in step_metrics["domains"].values()])
                        overall_normal_acc = np.mean([domain_data["normal_acc"] for domain_data in step_metrics["domains"].values()])
                        overall_easy_acc = np.mean([domain_data["easy_acc"] for domain_data in step_metrics["domains"].values()])
                        overall_total_avg_acc = np.mean([domain_data["total_avg_acc"] for domain_data in step_metrics["domains"].values()])
                        
                        step_metrics["overall"] = {
                            "hard_acc": overall_hard_acc,
                            "normal_acc": overall_normal_acc,
                            "easy_acc": overall_easy_acc,
                            "total_avg_acc": overall_total_avg_acc,
                            "sample_count": len(all_samples)
                        }
                    
                    # Add debug information
                    step_metrics["debug_info"] = {
                        "extraction_methods": dict(all_extraction_methods)
                    }
                    
                    all_metrics[step_number] = step_metrics
                    
                except json.JSONDecodeError:
                    print(f"Error: {filename} is not valid JSON")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    except FileNotFoundError:
        print(f"Error: Directory not found: '{directory_path}'")
        return {}
    
    return all_metrics




parser = argparse.ArgumentParser(description="Compute per-step accuracies from evaluation JSON files.")
parser.add_argument(
    "path",
    help="Path to the directory that contains the evaluation output files."
)
#parser.add_argument("--dataset", default="rmbench", help="Dataset name (default: %(default)s).")

args = parser.parse_args()
for dataset in ["rmbench", "rewardbench", "rmb"]:
    if dataset == "rmbench":
        metrics = compute_rmbench_metrics(args.path, dataset)
        print_rmbench_results(metrics)
    else:
        final_results = calculate_step_accuracies(args.path, dataset)
        print_eval_scores(dataset, final_results)

