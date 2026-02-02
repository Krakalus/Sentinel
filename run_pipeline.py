import torch
import pickle
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentinel_engine import SentinelEngine

# --- CONFIG ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LAYER = 31
# The Scientific Sweep Range
SIGMAS = [0.001, 0.01, 0.06, 0.1, 0.2] 
# Update this path to your actual CSV
DATA_PATH = "liars_bench_insider_trading.csv" 

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"[-] Loaded {len(df)} samples from {DATA_PATH}")
        return df['prompt'].tolist(), df['label'].tolist()
    except Exception as e:
        print(f"!! Data Load Error: {e}")
        print("!! WARNING: Generating SYNTHETIC PROOF DATA for pipeline verification !!")
        prompts = ["Honest prompt"]*20 + ["Deceptive prompt"]*20
        labels = [0]*20 + [1]*20
        return prompts, labels

def main():
    print("=== Phase 2 Scientific Pipeline ===")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    
    engine = SentinelEngine(model, tokenizer)
    prompts, labels = load_data()
    
    # 1. Run Surgical Sweep (Captures Energy Collapse)
    print("[-] Executing Surgical Sensitivity Sweep...")
    sweep_results = engine.run_surgical_sweep(prompts, labels, target_layer=LAYER, sigmas=SIGMAS)
    
    # 2. Run Benchmarks (Static vs Sentinel at optimal sigma)
    # This generates the Comparative ROC data
    print("[-] Benchmarking against Static Baseline...")
    benchmarks = engine.benchmark_baselines(sweep_results, labels, sigma_key=0.06)
    
    # 3. Compile Stability Data for Plotting
    stability_data = {'sigmas': SIGMAS, 'honest_means': [], 'deceptive_means': []}
    for s in SIGMAS:
        metrics = sweep_results[s]['metrics']
        energies = np.concatenate([m['energy_noisy'] for m in metrics])
        y = np.array(labels)
        
        stability_data['honest_means'].append(np.mean(energies[y==0]))
        stability_data['deceptive_means'].append(np.mean(energies[y==1]))

    # Save Payload
    final_results = {
        "benchmarks": benchmarks,
        "stability": stability_data,
        "labels": labels
    }
    
    with open("phase2_scientific_results.pkl", "wb") as f:
        pickle.dump(final_results, f)
        
    print("=== Success. Run 'plot_gold_standard.py' next. ===")

if __name__ == "__main__":
    main()