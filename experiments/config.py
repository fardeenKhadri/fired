"""
Configuration presets for the core pruning experiments.
"""

collapse_experiment = {
    "experiment_name": "push_to_collapse",
    "prune_method": "both",
    "retrain_after_prune": True,
    "prune_iterations": 10,
    "prune_percent_per_iter": 10.0,
    "gradient_accumulation_batches": 1,
    "target_sparsity_max": 100.0,
}

no_retrain_experiment = {
    "experiment_name": "no_retrain_comparison",
    "prune_method": "both",
    "retrain_after_prune": False,
    "prune_iterations": 5,
    "prune_percent_per_iter": 10.0,
    "gradient_accumulation_batches": 1,
    "target_sparsity_max": 50.0,
}

improved_gradient_experiment = {
    "experiment_name": "improved_gradient_estimation",
    "prune_method": "both",
    "retrain_after_prune": False,
    "prune_iterations": 7,
    "prune_percent_per_iter": 10.0,
    "gradient_accumulation_batches": 30,
    "target_sparsity_max": 70.0,
}
