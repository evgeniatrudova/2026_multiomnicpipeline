import os
import sys
import argparse
import subprocess
import shutil

def check_fgbio_env():
    if shutil.which("fgbio") is None:
        print("CRITICAL SERVER ERROR: 'fgbio' tool suite was not found.")
        sys.exit(1)

def run_consensus_dedup(input_dir, output_dir, threads, mode, sample, assay):
    print(f"--- Running Step 5: UMI Consensus Deduplication for {sample} [Mode: {mode.upper()}, Assay: {assay.upper()}] ---")
    
    alignment_results_dir = os.path.join(output_dir, "4_alignment_results")
    step_out_dir = os.path.join(output_dir, "5_consensus_results")
    os.makedirs(step_out_dir, exist_ok=True)

    input_bam = os.path.join(alignment_results_dir, f"{sample}_Aligned.sortedByCoord.out.bam")
    if not os.path.exists(input_bam):
        print(f"ERROR: Missing upstream genomic BAM alignment for {sample} at: {input_bam}")
        sys.exit(1)

    output_prefix = os.path.join(step_out_dir, sample)
    grouped_bam = f"{output_prefix}.grouped.bam"
    consensus_unmapped_bam = f"{output_prefix}.consensus.unmapped.bam"
    final_consensus_bam = f"{output_prefix}.consensus.bam"

    if mode == "mock":
        print(f"  > [SIMULATION] Simulating error-correction for: {sample}")
        with open(final_consensus_bam, "w") as f:
            f.write(f"MOCK ERROR-CORRECTED CONSENSUS BAM DATA FOR {sample}")
        return

    check_fgbio_env()
    
    # 1. Biological Stringency Router
    assay_lower = assay.lower()
    if assay_lower in ["mrna", "rna", "mirna"]:
        print("  > [ASSAY CONFIG] Transcriptomics detected. Setting min-reads to 1 to preserve expression abundance.")
        min_reads = "1"
    else:
        print("  > [ASSAY CONFIG] cfDNA Genomics detected. Setting min-reads to 3 to aggressively crush PCR errors.")
        min_reads = "3"
    
    group_cmd = [
        "fgbio", "GroupReadsByUmi",
        "--input", input_bam,
        "--output", grouped_bam,
        "--strategy", "paired",
        "--edits", "1"
    ]
    
    call_cmd = [
        "fgbio", "CallMolecularConsensusReads",
        "--input", grouped_bam,
        "--output", consensus_unmapped_bam,
        "--min-reads", min_reads,
        "--min-input-base-quality", "20"
    ]
    
    try:
        print(f"  > [PRODUCTION] Launching fgbio Grouping pipeline...")
        subprocess.run(group_cmd, check=True)
        print(f"  > [PRODUCTION] Launching Bayesian Consensus Caller...")
        subprocess.run(call_cmd, check=True)
    
        if os.path.exists(grouped_bam):
            os.remove(grouped_bam)
            
        shutil.copy(consensus_unmapped_bam, final_consensus_bam)
    
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL ERROR: fgbio deduplication crashed on {sample}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: fgbio UMI Deduplication")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["mock", "prod"], default="prod")
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--assay", type=str, required=False, default="cfdna")
    
    args = parser.parse_args()
    run_consensus_dedup(args.input, args.output, args.threads, args.mode, args.sample, args.assay)