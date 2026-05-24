import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_vaf_distribution(df, sample_name, output_plot):
    if df.empty:
        print(f"WARNING: No active variants discovered for {sample_name}. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(df['VAF'] * 100, bins=30, color='salmon', edgecolor='black', alpha=0.7, log=True)
    plt.axvline(x=0.1, color='red', linestyle='--', linewidth=1.5, label='Ultra-low VAF Boundary (0.1%)')
    
    plt.title(f"Somatic Mutation Profile (VAF Spectrum): {sample_name}", fontsize=14)
    plt.xlabel("Variant Allele Frequency (VAF %)", fontsize=12)
    plt.ylabel("Variant Count (Log Scale)", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()
    print(f"  > SUCCESS: VAF profile dashboard generated at {output_plot}")

def run_variant_visualization(input_dir, output_dir, threads, mode, sample, assay):
    print(f"--- Running Step 9: Somatic Variant VAF Spectrum Visualization for {sample} [Mode: {mode.upper()}] ---")
    
    # BIOLOGICAL ROUTING: VAF is strictly a genomic (DNA) concept.
    if assay.lower() in ["mrna", "mirna", "rna"]:
        print(f"  > [ASSAY CONFIG] {assay.upper()} detected. RNA variants exhibit Allele-Specific Expression (ASE), not true VAF.")
        print(f"  > Gracefully bypassing genomic VAF histogram generation.")
        return

    variant_results_dir = os.path.join(output_dir, "8_variant_results")
    step_out_dir = os.path.join(output_dir, "9_variant_plots")
    os.makedirs(step_out_dir, exist_ok=True)

    vcf_path = os.path.join(variant_results_dir, f"{sample}_final_somatic_variants.vcf.gz")
    if not os.path.exists(vcf_path):
        print(f"ERROR: Upstream somatic variant call record not found for {sample} at: {vcf_path}")
        sys.exit(1)

    plot_out = os.path.join(step_out_dir, f"{sample}_vaf_sanity_check.png")

    if mode == "mock":
        print(f"  > [SIMULATION] Parsing mutations and rendering profile coordinates for: {sample}")
        np.random.seed(42)
        mock_vafs = np.random.exponential(scale=0.005, size=45)
        mock_vafs = mock_vafs[mock_vafs < 0.05]
        
        df_mock = pd.DataFrame({
            'VAF': mock_vafs,
            'DP': np.random.randint(800, 2500, size=len(mock_vafs)),
            'ID': [f"chr17:{p}" for p in np.random.randint(7500000, 7600000, size=len(mock_vafs))]
        })
        plot_vaf_distribution(df_mock, sample, plot_out)

    elif mode == "prod":
        print(f"  > [PRODUCTION] Parsing true VCF stream for: {sample}")
        import pysam
        vaf_records = []
        try:
            vcf_file = pysam.VariantFile(vcf_path)
            for record in vcf_file:
                if "PASS" in record.filter:
                    af = record.samples[0]['AF'][0]
                    dp = record.samples[0]['DP']
                    vaf_records.append({'VAF': af, 'DP': dp, 'ID': f"{record.chrom}:{record.pos}"})
            
            df_real = pd.DataFrame(vaf_records)
            plot_vaf_distribution(df_real, sample, plot_out)
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to parse native VCF stream on {sample}: {e}")
            sys.exit(1)

    print(f"\nStep 9 Processing Complete. Diagnostics available in: {step_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 9: Somatic Variant Allele Frequency Profiler")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["mock", "prod"], default="prod")
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--assay", type=str, required=False, default="cfdna")
    
    args = parser.parse_args()
    run_variant_visualization(args.input, args.output, args.threads, args.mode, args.sample, args.assay)