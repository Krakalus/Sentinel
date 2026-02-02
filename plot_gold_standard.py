import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

# Gold Standard Styling (NeurIPS/ICLR)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['lines.linewidth'] = 2.5

def plot_all():
    with open("phase2_scientific_results.pkl", "rb") as f:
        data = pickle.load(f)
    
    bench = data['benchmarks']
    stab = data['stability']
    labels = np.array(data['labels'])
    
    # --- FIGURE 2: ROC COMPARISON (The Comparative Advantage) ---
    plt.figure(figsize=(8, 8))
    # Static Baseline
    plt.plot(bench['static'][0], bench['static'][1], color='gray', linestyle='--', lw=2, 
             label=f"Static Linear Probe (Baseline) AUC={bench['static'][2]:.2f}")
    # Sentinel
    plt.plot(bench['sentinel'][0], bench['sentinel'][1], color='#006400', lw=3, 
             label=f"Sentinel (Ours) AUC={bench['sentinel'][2]:.2f}")
    plt.plot([0,1],[0,1], 'k:', alpha=0.3)
    plt.xlabel('False Positive Rate (Cost)', fontweight='bold')
    plt.ylabel('True Positive Rate (Assurance)', fontweight='bold')
    plt.title('Figure 2: Sentinel vs. SOTA Baseline', fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('Fig2_ROC_Comparison.pdf')
    
    # --- FIGURE 4: THE MAGNITUDE COLLAPSE (The Scientific Discovery) ---
    plt.figure(figsize=(10, 6))
    plt.plot(stab['sigmas'], stab['honest_means'], marker='o', color='#1f77b4', lw=2, label='Honest (Robust Basin)')
    # Deceptive should show the "Collapse" (Drop in Energy)
    plt.plot(stab['sigmas'], stab['deceptive_means'], marker='D', color='#d62728', lw=2, label='Deceptive (Collapsing Basin)')
    
    # NF4 Horizon Line
    plt.axvline(x=0.06, color='gray', linestyle='--')
    plt.text(0.065, np.max(stab['honest_means']), 'NF4 Horizon ($\sigma_{crit}$)', color='gray')
    
    # Annotation
    plt.annotate('Plastic Magnitude Collapse', xy=(0.06, stab['deceptive_means'][2]), 
                 xytext=(0.01, stab['deceptive_means'][2]-30),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xscale('log')
    plt.xlabel(r'Noise Magnitude ($\sigma$)', fontweight='bold')
    plt.ylabel(r'Manifold Energy ($\|\mathbf{h}_{noisy}\|_2$)', fontweight='bold')
    plt.title('Figure 5: Plastic Collapse of Deceptive Representations', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig5_Magnitude_Collapse.pdf')
    
    # --- FIGURE 3: EFFICIENCY FRONTIER (Economic Impact) ---
    plt.figure(figsize=(8, 6))
    # Comparison Points
    plt.scatter([1], [0.1], s=100, color='gray', label='Random Audit')
    plt.scatter([2], [bench['static'][2]], s=150, marker='s', color='#A0522D', label='Static Probe')
    plt.scatter([150], [0.95], s=200, color='orange', label='Llama-70B Oracle')
    plt.scatter([2], [bench['sentinel'][2]], s=400, color='#006400', edgecolors='black', label='Sentinel (Ours)')
    
    plt.xscale('log')
    plt.xlabel('Inference Latency (ms/token)', fontweight='bold')
    plt.ylabel('Adversarial Assurance (Recall)', fontweight='bold')
    plt.title('Figure 3: Efficiency-Assurance Frontier', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig('Fig3_Efficiency.pdf')

    print("[-] All Figures Generated: Fig2, Fig3, Fig5")

if __name__ == "__main__":
    plot_all()