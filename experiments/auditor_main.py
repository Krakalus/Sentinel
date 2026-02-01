import sys
import os
import traceback

# ==============================================================================
# [CRITICAL] PATH BOOTSTRAP: Solves "ModuleNotFoundError" on Linux/H100
# ==============================================================================
current_file = os.path.abspath(__file__)
exp_dir = os.path.dirname(current_file)
project_root = os.path.dirname(exp_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==============================================================================

import argparse
import torch
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import gc
from datetime import datetime
from rich.console import Console
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Sentinel Imports
from experiments.tasks.liars_bench_adapter import LiarsBenchAdapter
from experiments.utils.harvester import ActivationHarvester
from experiments.utils.deception_probe import OrthogonalProbe
from experiments.utils.pareto import ParetoCombiner

console = Console()

def save_forensic_dashboard(layer_data, filename="results/sentinel_forensic_dashboard.png"):
    """
    Generates a High-Fidelity Forensic Dashboard.
    Top: DNA-Style Heatmap of Deception Signal.
    Bottom: Component Analysis (Raw vs. Purified).
    """
    full_path = os.path.join(project_root, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    layers = sorted(layer_data.keys())
    if not layers:
        console.print("[yellow]Warning: No layer data to plot.[/yellow]")
        return

    # Extract Purified Scores for the Heatmap
    scores = [layer_data[l]['purified'] for l in layers]
    df_heat = pd.DataFrame([scores], columns=layers)
    
    # Setup Canvas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 3]})
    sns.set_theme(style="whitegrid")

    # 1. Heatmap Strip
    sns.heatmap(df_heat, cmap="magma", ax=ax1, cbar=False, yticklabels=False, xticklabels=False)
    ax1.set_title("SENTINEL THREAT SIGNATURE (HEATMAP)", fontsize=12, fontweight='bold', letterspacing=2)
    
    # 2. Detailed Signal Analysis
    raw_scores = [layer_data[l]['raw'] for l in layers]
    purified_scores = [layer_data[l]['purified'] for l in layers]
    
    sns.lineplot(x=layers, y=raw_scores, ax=ax2, color="grey", linestyle="--", label="Raw Intent (Contaminated)")
    sns.lineplot(x=layers, y=purified_scores, ax=ax2, marker="o", linewidth=3, color="#d62728", label="Purified Signal (Orthogonal)")
    ax2.fill_between(layers, purified_scores, alpha=0.1, color="#d62728")
    
    # Highlight Peak
    if purified_scores:
        max_score = max(purified_scores)
        max_layer = layers[purified_scores.index(max_score)]
        ax2.axvline(x=max_layer, color='green', linestyle=':', label=f"Deception Horizon (L{max_layer})")
    
    ax2.set_title("ORTHOGONAL SUBSPACE PROJECTION ANALYSIS", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Transformer Layer Depth", fontsize=10)
    ax2.set_ylabel("Signal Magnitude (σ)", fontsize=10)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.close()
    console.print(f"[bold green]✔ Forensic Artifact Saved:[/bold green] {full_path}")

def main():
    parser = argparse.ArgumentParser(description="Sentinel Sleeper Agent Auditor")
    parser.add_argument("--mode", type=str, default="SENTINEL_AUDIT")
    parser.add_argument("--precision", type=str, default="BF16")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct") 
    parser.add_argument("--scan_start", type=int, default=10)
    parser.add_argument("--scan_end", type=int, default=24)
    
    args = parser.parse_args()

    console.print("\n[bold cyan]==================================================[/bold cyan]")
    console.print("[bold cyan]   SENTINEL: HYBRID ORTHOGONAL PURIFICATION ENGINE[/bold cyan]")
    console.print(f"[bold cyan]   Target: {args.model} | Precision: {args.precision}[/bold cyan]")
    console.print("[bold cyan]==================================================\n[/bold cyan]")

    # 1. SETUP PRECISION
    if args.precision == "BF16" and torch.cuda.is_available():
        dtype = torch.bfloat16
        console.print("[yellow][!] Enforcing Bfloat16 to prevent Manifold Corruption.[/yellow]")
    else:
        dtype = torch.float32
        console.print("[dim][!] Using Float32.[/dim]")

    console.print(f"[bold]Loading Model...[/bold]")
    
    # Robust Tokenizer Loading
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    except Exception as e:
        console.print(f"[bold red]FATAL: Tokenizer load failed.[/bold red]")
        console.print(f"Error: {e}")
        return

    # 2. TWIN-STREAM MODEL LOADING (Hypothesis 1: Deception Horizon)
    console.print(f"[bold]Initializing Twin-Stream Architecture...[/bold]")

    # Stream A: Reference (High Precision - BF16)
    console.print("   [1/2] Loading Reference Stream (BF16)...")
    model_hi = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    model_hi.eval()

    # Stream B: Stress Test (Low Precision - NF4)
    # This implements "Quantization as Structural Stress"
    console.print("   [2/2] Loading Stress Stream (NF4)...")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_lo = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=nf4_config,
        device_map="auto"
    )
    model_lo.eval()

    # 3. DATA INGESTION
    console.print(f"[bold]Ingesting Liars Bench...[/bold]") 

    adapter = LiarsBenchAdapter()
    data_pairs = adapter.load_dataset()
    
    texts_U = [d['text_unmonitored'] for d in data_pairs]
    texts_M = [d['text_monitored'] for d in data_pairs]
    labels  = torch.tensor([d['label'] for d in data_pairs])

    if labels.sum() < 2:
        console.print("[bold red]CRITICAL: Insufficient deceptive samples. Run 'python generate_audit_data.py' first![/bold red]")
        return

    # 4. FORENSIC SWEEP
    layer_metrics = {} 
    best_score = -1.0
    best_probe = None
    best_acts_U = None
    
    if hasattr(model_hi.config, "num_hidden_layers"):
        args.scan_end = min(args.scan_end, model_hi.config.num_hidden_layers - 1)

    console.print(f"\n[bold]Initiating Forensic Sweep:[/bold] Layers {args.scan_start} to {args.scan_end}")
    
    for layer_idx in track(range(args.scan_start, args.scan_end + 1), description="Scanning Layers..."):

        # Harvest High Precision (Reference)
        harvester_hi = ActivationHarvester(model_hi, tokenizer, layer_idx)
        acts_U_hi = harvester_hi.harvest(texts_U, batch_size=8, verbose=False)
        acts_M_hi = harvester_hi.harvest(texts_M, batch_size=8, verbose=False)      

        # Harvest Low Precision (Stress)
        harvester_lo = ActivationHarvester(model_lo, tokenizer, layer_idx)
        acts_U_lo = harvester_lo.harvest(texts_U, batch_size=8, verbose=False)
        acts_M_lo = harvester_lo.harvest(texts_M, batch_size=8, verbose=False)

        # --- CORE MECHANIC: DIFFERENTIAL PROBING ---
        # Calculate Quantization Drift: Delta = High - Low
        acts_U_diff = acts_U_hi - acts_U_lo
        acts_M_diff = acts_M_hi - acts_M_lo       

        # Probe
        mask_honest = (labels == 0)
        mask_deceptive = (labels == 1)
        
        probe = OrthogonalProbe()
        probe.fit(acts_U_diff[mask_honest], acts_U_diff[mask_deceptive], acts_M_diff[mask_honest], verbose=False)
        
        # Metrics
        raw_norm = 0.0
        if probe.mu_DU is not None:
            raw_norm = (probe.mu_DU - probe.mu_HU).norm().item()
            
        separation = probe.score(acts_U_diff[mask_deceptive]).mean()
        
        layer_metrics[layer_idx] = {
            'raw': raw_norm,
            'purified': float(separation)
        }
        
        if separation > best_score:
            best_score = separation
            best_probe = probe
            best_acts_U = acts_U_diff            

        # VRAM SAFETY
        del harvester_hi, harvester_lo, acts_U_hi, acts_U_lo, acts_U_diff, acts_M_hi, acts_M_lo, acts_M_diff, probe
        torch.cuda.empty_cache()
        gc.collect()

    # 5. RESULTS
    if not layer_metrics:
        console.print("[bold red]No layers scanned.[/bold red]")
        return

    peak_layer = max(layer_metrics, key=lambda x: layer_metrics[x]['purified'])
    console.print(f"\n[bold green]Scan Complete.[/bold green] Peak Horizon: L{peak_layer}")
    
    dashboard_path = os.path.join(project_root, "results", "sentinel_forensic_dashboard.png")
    save_forensic_dashboard(layer_metrics, filename=dashboard_path)
    
    # Pareto
    if best_probe:
        console.print("\n[bold]Calculating Pareto Frontier...[/bold]")
        clean_scores = best_probe.score(best_acts_U)
        exp_scores = clean_scores + np.random.normal(0, 0.2, len(clean_scores))
        combiner = ParetoCombiner()
        combiner.fit_optimization(clean_scores, exp_scores, labels.numpy(), budget_constraint=15.0)

    console.print("\n[bold cyan]=== SENTINEL AUDIT COMPLETE ===[/bold cyan]")

if __name__ == "__main__":
    main()
