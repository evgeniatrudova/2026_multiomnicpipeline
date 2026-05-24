import os
import sys
import argparse
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# FRAGMENTOMICS & MULTI-OMICS SEQUENCE PROFILING
# ==============================================================================
# - cfDNA: Tracks 4-mer nucleotide motifs at the 5' cleavage ends. Aberrant 
#   nuclease cutting in cancer shifts these frequencies (e.g., 'CCCA').
# - miRNA: Validates that sequences are ~22bp long. If sequences average 100bp, 
#   the library prep failed. Biologically, tracks the 1st nucleotide bias 
#   (Argonaute proteins strongly prefer 5'-Uridine/Thymine).
# ==============================================================================

def plot_motifs(motif_dict, sample_name, plot_path, assay):
    """Generates the bar chart for motif or terminal nucleotide frequencies."""
    df = pd.DataFrame.from_dict(motif_dict, orient='index', columns=['frequency'])
    
    plt.figure(figsize=(10, 6))
    
    if assay.lower() == "mirna":
        df['frequency'].plot(kind='bar', color='coral')
        plt.title(f"5' Terminal Nucleotide Bias (miRNA): {sample_name}", fontsize=14)
        plt.xlabel("1st Nucleotide Position", fontsize=12)
    else:
        df['frequency'].plot(kind='bar', color='orchid')
        plt.title(f"Top Fragment End-Motifs (cfDNA): {sample_name}", fontsize=14)
        plt.xlabel("4-mer Motif Sequence", fontsize=12)
        
    plt.ylabel("Normalized Frequency (%)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def run_fragmentomics(input_dir, output_dir, threads, mode, sample, assay):
    print(f"--- Running Step 7: Sequence Profiling for {sample} [Mode: {mode.upper()}, Assay: {assay.upper()}] ---")
    
    consensus_dir = os.path.join(output_dir, "5_consensus_results")
    step_out_dir = os.path.join(output_dir, "7_fragmentomics_results")
    os.makedirs(step_out_dir, exist_ok=True)

    input_bam = os.path.join(consensus_dir, f"{sample}.consensus.bam")
    if not os.path.exists(input_bam):
        print(f"ERROR: Upstream consensus BAM file not found for {sample} at: {input_bam}")
        sys.exit(1)

    plot_path = os.path.join(step_out_dir, f"{sample}_sequence_profile.png")
    assay_lower = assay.lower()

    # ----------------------------------------------------------------------
    # MOCK / SIMULATION MODE
    # ----------------------------------------------------------------------
    if mode == "mock":
        print(f"  > [SIMULATION] Computing mock sequence bias matrix for sample: {sample}")
        
        if assay_lower == "mirna":
            print("  > [miRNA QC] Validating simulated read lengths: Average 22.1bp [PASS]")
            mock_motifs = {'T': 78.5, 'A': 12.1, 'C': 5.4, 'G': 4.0} # Massive T/U preference
        else:
            mock_motifs = {
                'CCCA': 4.2, 'AAAA': 3.8, 'TATA': 2.9, 'GGGG': 2.1, 
                'CTAG': 1.8, 'AGCT': 1.7, 'TGCA': 1.5, 'CGCG': 1.2
            }
            
        plot_motifs(mock_motifs, sample, plot_path, assay)

    # ----------------------------------------------------------------------
    # PRODUCTION MODE
    # ----------------------------------------------------------------------
    elif mode == "prod":
        # We import pysam strictly inside production mode so Windows users don't crash!
        import pysam 
        
        motif_counts = Counter()
        read_lengths = []
        total_fragments = 0
        max_reads = 1000000
        
        print(f"  > [PRODUCTION] Opening high-speed pysam alignment reader thread for: {sample}")
        try:
            with pysam.AlignmentFile(input_bam, "rb") as bam:
                for read in bam:
                    if read.is_unmapped or read.is_duplicate or read.is_secondary:
                        continue
                        
                    sequence = read.query_sequence
                    if sequence:
                        # Track lengths for strict QC validation
                        read_lengths.append(len(sequence))
                        
                        # Extract appropriate biological sequence signature
                        if assay_lower == "mirna" and len(sequence) >= 1:
                            motif = sequence[0] # 5' Terminal Nucleotide
                            motif_counts[motif] += 1
                        elif len(sequence) >= 4:
                            motif = sequence[:4] # 4-mer End Motif
                            motif_counts[motif] += 1
                            
                        total_fragments += 1
                        
                    if total_fragments >= max_reads:
                        break
        
            # --- STRICT LIBRARY PREP VALIDATION ---
            if read_lengths:
                avg_len = np.mean(read_lengths)
                print(f"  > [QC METRIC] Average Read Sequence Length: {avg_len:.1f} bp")
                
                if assay_lower == "mirna":
                    if avg_len > 40:
                        print(f"  > ❌ CRITICAL WARNING: {sample} failed miRNA length validation.")
                        print(f"  > Expected ~22bp, detected {avg_len:.1f}bp. Library preparation likely failed or adapter trimming was bypassed.")
                    else:
                        print("  > ✅ miRNA sequence length validation passed.")
            
            # --- DOWNSTREAM PLOTTING ---
            if total_fragments > 0:
                if assay_lower == "mirna":
                    # For single nucleotides, we map the top 4
                    top_motifs = dict(motif_counts.most_common(4))
                else:
                    # For cfDNA 4-mers, we map the top 8
                    top_motifs = dict(motif_counts.most_common(8))
                    
                normalized_motifs = {k: (v / total_fragments) * 100 for k, v in top_motifs.items()}
                plot_motifs(normalized_motifs, sample, plot_path, assay)
            else:
                print(f"WARNING: No valid fragments found in {sample} to profile.")
                
        except Exception as e:
            print(f"CRITICAL SYSTEM ERROR: pysam failed to extract motifs from {sample}: {e}")
            sys.exit(1)

    print(f"Step 7 Complete. Sequence profiles compiled in: {step_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Fragmentomics Motif Profiler")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["mock", "prod"], default="prod", help="Execution mode")
    parser.add_argument("--sample", type=str, required=True, help="Explicit Sample ID to process")
    parser.add_argument("--assay", type=str, required=False, default="cfdna", help="Target genetic material")
    
    args = parser.parse_args()
    run_fragmentomics(args.input, args.output, args.threads, args.mode, args.sample, args.assay)