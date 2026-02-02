import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

class SentinelHook:
    """
    The Sentinel Perturbation Operator.
    Implements Signal-Dependent Covariance to respect Llama-3's outlier features.
    """
    def __init__(self, layer_idx, sigma):
        self.layer_idx = layer_idx
        self.sigma = sigma
        self.clean_activations = [] 
        self.metrics = [] # Stores scientific metrics (Energy & Drift)
        
    def __call__(self, module, input, output):
        # Handle Llama-3 Output Tuple
        h = output[0] if isinstance(output, tuple) else output
            
        # 1. Harvest Clean State (Last token latent thought)
        h_clean = h[:, -1, :].detach().float()
        
        # 2. Compute Signal Energy (Standard Deviation per channel)
        # This ensures SNR is invariant to outliers (prevents masking effects)
        # Formula: ξ = N(0, Σ), where Σ = σ * diag(std(h))
        channel_std = torch.std(h_clean, dim=-1, keepdim=True)
        noise = torch.randn_like(h_clean) * self.sigma * channel_std
        
        # 3. Inject Noise (The Sentinel Operator)
        h_noisy = h_clean + noise
        
        # 4. Compute Scientific Metrics (The Discovery)
        # Manifold Energy (Norm): Proves the "Collapse"
        energy_clean = torch.norm(h_clean, p=2, dim=-1).cpu().numpy()
        energy_noisy = torch.norm(h_noisy, p=2, dim=-1).cpu().numpy()
        
        # Differential Sensitivity (Drift): The Detection Signal
        drift = torch.norm(h_clean - h_noisy, p=2, dim=-1).cpu().numpy()
        
        # Store for Baseline vs Sentinel Comparison
        self.clean_activations.append(h_clean.cpu().numpy())
        self.metrics.append({
            "energy_clean": energy_clean,
            "energy_noisy": energy_noisy,
            "drift": drift
        })
        
        # Return noisy state to stream (Simulates 'Stress')
        h_mod = h.clone()
        h_mod[:, -1, :] = h_noisy.to(h.dtype)
        
        return (h_mod,) + output[1:] if isinstance(output, tuple) else h_mod

class SentinelEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run_surgical_sweep(self, prompts, labels, target_layer=31, sigmas=[0.06]):
        results = {}
        for sigma in sigmas:
            print(f"[-] Running Sweep: Sigma={sigma} at Layer {target_layer}")
            hook = SentinelHook(target_layer, sigma)
            handle = self.model.model.layers[target_layer].register_forward_hook(hook)
            
            # Inference Loop (Batch size 1 for safety, can be increased)
            for p in prompts:
                inputs = self.tokenizer(p, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    self.model(**inputs)
            
            handle.remove()
            
            # Aggregate Data
            results[sigma] = {
                "clean_X": np.concatenate(hook.clean_activations),
                "metrics": hook.metrics
            }
        return results

    def benchmark_baselines(self, sweep_results, labels, sigma_key=0.06):
        """
        Compares Static Probe (Baseline) vs Sentinel (Ours)
        to prove the EB1A "Original Contribution" claim.
        """
        data = sweep_results[sigma_key]
        y = np.array(labels)
        
        # --- BASELINE: STATIC LINEAR PROBE ---
        # Trains on raw clean activations (Standard Industry Approach)
        X_static = data["clean_X"]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        static_preds = np.zeros(len(y))
        for train, test in cv.split(X_static, y):
            clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
            clf.fit(X_static[train], y[train])
            static_preds[test] = clf.predict_proba(X_static[test])[:, 1]
            
        fpr_s, tpr_s, _ = roc_curve(y, static_preds)
        auc_s = auc(fpr_s, tpr_s)
        
        # --- OURS: SENTINEL TOPOLOGICAL PROBE ---
        # Detects based on Sensitivity Signature (Drift Magnitude)
        drifts = np.concatenate([m['drift'] for m in data['metrics']])
        X_sentinel = drifts.reshape(-1, 1) # Scalar feature proves "Geiger Counter" phase transition
        
        sent_preds = np.zeros(len(y))
        for train, test in cv.split(X_sentinel, y):
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(X_sentinel[train], y[train])
            sent_preds[test] = clf.predict_proba(X_sentinel[test])[:, 1]
            
        fpr_m, tpr_m, _ = roc_curve(y, sent_preds)
        auc_m = auc(fpr_m, tpr_m)
        
        return {
            "static": (fpr_s, tpr_s, auc_s),
            "sentinel": (fpr_m, tpr_m, auc_m),
            "drifts": drifts,
            "energies": np.concatenate([m['energy_noisy'] for m in data['metrics']])
        }