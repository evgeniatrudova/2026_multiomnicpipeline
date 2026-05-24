import os
import sys
import argparse
import subprocess
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# FRAGMENT TOGGLE & MULTI-OMICS ROUTING
# ==============================================================================
# Different assays have vastly different expected fragment length distributions:
# - cfDNA: Apoptotic nuclease cleavage creates peaks at ~167bp and ~145bp.
# - miRNA: Mature microRNAs are incredibly short and stable (~22bp).
# - mRNA: Heavily dependent on chemical/mechanical library fragmentation (often ~300bp).
#
# - MOCK MODE: Synthesizes mathematically appropriate distributions for the selected assay.
# - PRODUCTION MODE: Dynamically adjusts awk thresholds to prevent throwing out
#   valid biological reads (e.g., dropping 22bp miRNAs because of a 30bp cfDNA filter).
# ==============================================================================

def check_dependencies():
    """Verifies that samtools is accessible within the server's execution environment."""
    if shutil.which("samtools") is None:
        print("CRITICAL SERVER ERROR: 'samtools' binary utility not discovered in active PATH.")
        sys.exit(1)

def plot_fragment_distribution(tsv_file, sample_name, plot_path, assay):
    """Generates a high-resolution Kernel Density Estimate (KDE) plot of the fragment sizes."""
    print(f"  > Generating {assay.upper()} Fragmentomics Visualisation for: {sample_name}...")
    
    data = pd.read_csv(tsv_file, header=None, names=['size'])
    if data.empty or len(data) < 5:
         print(f"WARNING: Skipping visualization for {sample_name} - Insufficient data points.")
         return

    plt.figure(figsize=(10, 6))
    # Use native pandas layout tracking to follow constraints smoothly
    ax = data['size'].plot(kind='kde', color='teal', linewidth=2)
    
    # Dynamically inject assay-specific biological markers and constraints
    assay = assay.lower()
    if assay == "mirna":
        x_min, x_max = 10, 50
        plt.axvline(x=22, color='navy', linestyle='--', alpha=0.8, label='Mature miRNA (~22bp)')
        plt.title(f"miRNA Length Distribution Profile: {sample_name}", fontsize=14)
        
    elif assay in ["mrna", "rna"]:
        x_min, x_max = 50, 1000
        plt.axvline(x=300, color='navy', linestyle='--', alpha=0.6, label='Expected Target Library Size (~300bp)')
        plt.title(f"RNA-Seq Library Insert Size Profile: {sample_name}", fontsize=14)
        
    else: # cfDNA default
        x_min, x_max = 30, 500
        plt.axvline(x=145, color='red', linestyle='--', alpha=0.6, label='Tumor Peak Core (145bp)')
        plt.axvline(x=167, color='navy', linestyle='--', alpha=0.6, label='Healthy Peak Core (167bp)')
        plt.title(f"cfDNA Fragment Size Distribution Profile: {sample_name}", fontsize=14)

    plt.xlabel("Insert Size / Template Length (bp)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"  > SUCCESS: Structural Integrity profile visual saved at: {plot_path}")

def run_structural_sanity(input_dir, output_dir, threads, mode, sample, assay):
    print(f"--- Running Step 6: Samtools Fragment Size Extraction for {sample} [Mode: {mode.upper()}, Assay: {assay.upper()}] ---")
    
    consensus_dir = os.path.join(output_dir, "5_consensus_results")
    step_out_dir = os.path.join(output_dir, "6_structural_checks")
    os.makedirs(step_out_dir, exist_ok=True)

    # Snakemake-ready: Explicitly define the expected input BAM for this sample
    input_bam = os.path.join(consensus_dir, f"{sample}.consensus.bam")

    if not os.path.exists(input_bam):
        print(f"ERROR: Upstream molecular consensus BAM file not found for {sample} at: {input_bam}")
        sys.exit(1)

    temp_tsv = os.path.join(step_out_dir, f"{sample}_temp_sizes.tsv")
    final_plot = os.path.join(step_out_dir, f"{sample}_fragment_profile.png")

    # Determine assay-specific length boundaries for filtering
    assay_lower = assay.lower()
    if assay_lower == "mirna":
        min_len, max_len = 10, 50
    elif assay_lower in ["mrna", "rna"]:
        min_len, max_len = 50, 1000
    else: # cfdna
        min_len, max_len = 30, 500

    # ----------------------------------------------------------------------
    # MOCK / SIMULATION MODE
    # ----------------------------------------------------------------------
    if mode == "mock":
        print(f"  > [SIMULATION] Streaming mock insert sizes and synthesizing {assay.upper()} curve for: {sample}")
        np.random.seed(42)
        
        if assay_lower == "mirna":
            # Razor-sharp normal distribution for miRNAs
            simulated_data = np.random.normal(loc=22, scale=1.5, size=5000)
        elif assay_lower in ["mrna", "rna"]:
            # Broad standard distribution for mechanical/enzymatic RNA fragmentation
            simulated_data = np.random.normal(loc=300, scale=60, size=8000)
        else: # cfdna
            # Bimodal healthy/tumor mix
            healthy_sizes = np.random.normal(loc=167, scale=25, size=7000)
            tumor_sizes = np.random.normal(loc=145, scale=20, size=3000)
            simulated_data = np.concatenate([healthy_sizes, tumor_sizes])
            
        # Apply the exact bounds we would use in production
        simulated_data = simulated_data[(simulated_data >= min_len) & (simulated_data <= max_len)]
        
        df_mock = pd.DataFrame(simulated_data.astype(int))
        df_mock.to_csv(temp_tsv, index=False, header=False)
        
    # ----------------------------------------------------------------------
    # PRODUCTION MODE
    # ----------------------------------------------------------------------
    elif mode == "prod":
        check_dependencies()
        # Dynamically inject the correct length bounds into the awk extraction command
        extract_cmd = f"samtools view {input_bam} | awk '{{print ($9<0?-$9:$9)}}' | awk '$1>={min_len} && $1<={max_len}' > {temp_tsv}"
        
        print(f"  > [PRODUCTION] Launching samtools binary extraction thread for: {sample}")
        try:
            subprocess.run(extract_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"CRITICAL SYSTEM ERROR: Samtools view stream broken on sample {sample}: {e}")
            sys.exit(1)

    # Execute downstream plotting module regardless of mode
    if os.path.exists(temp_tsv):
        plot_fragment_distribution(temp_tsv, sample, final_plot, assay)
        # Remove intermediate raw numbers to keep filesystem lightweight
        os.remove(temp_tsv)

    print(f"\nStep 6 Structural Analysis Complete. Plots written to: {step_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: Samtools Structural Fragmentomics Metric Profiler")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["mock", "prod"], default="prod", help="Execution mode")
    
    # Snakemake-ready arguments passed dynamically from step0
    parser.add_argument("--sample", type=str, required=True, help="Explicit Sample ID to process")
    parser.add_argument("--assay", type=str, required=False, default="cfdna", help="Target genetic material")
    
    args = parser.parse_args()
    run_structural_sanity(args.input, args.output, args.threads, args.mode, args.sample, args.assay)