import os
import sys
import argparse
import subprocess
import shutil

# ==============================================================================
# TRIMMING
# ==============================================================================
# Downstream UMI tools (fgbio/zUMIs) require Read 1 (R1) to remain completely 
# untrimmed to preserve exact barcode lengths. 
#
# We use asymmetric trimming (`--pair-filter=any`, `-m 20`) alongside only the 
# `-A` flag (which targets R2) to trim adapters/dimers from R2, while safely 
# leaving R1 intact. Do NOT add an `-a` flag for R1.
# Verify R1 length was preserved (both commands must output the exact same number):
#   zcat raw_data/sample1_R1_001.fastq.gz | head -n 2 | tail -n 1 | wc -c
#   zcat 1_trimmed_reads/sample1_R1_trimmed.fastq.gz | head -n 2 | tail -n 1 | wc -c
# ==============================================================================

def run_trimming(input_dir, output_dir, threads, mode, sample, assay):
    print(f"--- Running Step 1: Asymmetric Trimming for {sample} [Mode: {mode.upper()}, Assay: {assay.upper()}] ---")
    
    if mode == "prod" and shutil.which("cutadapt") is None:
        print("CRITICAL ERROR: 'cutadapt' not found. Is it installed in your Conda env?")
        sys.exit(1)

    step_out_dir = os.path.join(output_dir, "1_trimmed_reads")
    os.makedirs(step_out_dir, exist_ok=True)

    # NO glob. Explicitly target the exact file for this sample.
    r1_input = os.path.join(input_dir, f"{sample}_R1_001.fastq.gz")
    r2_input = os.path.join(input_dir, f"{sample}_R2_001.fastq.gz")

    if not os.path.exists(r1_input):
        print(f"ERROR: Missing input R1 file for {sample}. Expected at: {r1_input}")
        sys.exit(1)
        
    if not os.path.exists(r2_input):
        print(f"ERROR: Missing input R2 file for {sample}. Expected at: {r2_input}")
        sys.exit(1)

    # Construct explicit output file names
    r1_out = os.path.join(step_out_dir, f"{sample}_R1_trimmed.fastq.gz")
    r2_out = os.path.join(step_out_dir, f"{sample}_R2_trimmed.fastq.gz")
    log_path = os.path.join(step_out_dir, f"{sample}_trim_report.txt")

    adapter_r2 = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGTAGATCTCGGTGGTCGCCGTATCATT"

    # CRITICAL MULTI-OMICS FIX: Adjust length filter for ultra-short miRNA
    min_length = "15" if assay.lower() == "mirna" else "20"

    # ----------------------------------------------------------------------
    # MOCK / SIMULATION MODE
    # ----------------------------------------------------------------------
    if mode == "mock":
        print(f"  > [SIMULATION] Generating mock trimmed files for {sample}")
        for mock_file in [r1_out, r2_out, log_path]:
            with open(mock_file, 'w') as f:
                f.write("MOCK TRIMMED DATA")
        return # Exit the function early in mock mode

    # ----------------------------------------------------------------------
    # PRODUCTION MODE
    # ----------------------------------------------------------------------
    command = [
        "cutadapt",
        "-j", str(threads),              
        "-A", adapter_r2,        
        "-q", "20",              
        "-m", min_length,   # <--- Dynamic parameter injected here      
        "--pair-filter=any",          
        "-o", r1_out,            
        "-p", r2_out,            
        r1_input, r2_input       
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        with open(log_path, "w") as f:
            f.write(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Cutadapt failed on {sample}.")
        for partial in [r1_out, r2_out, log_path]:
            if os.path.exists(partial):
                os.remove(partial)
        sys.exit(1)

    print(f"\nStep 1 Complete. Trimmed files saved to: {step_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Cutadapt Trimming")
    parser.add_argument("-i", "--input", required=True, help="Raw data directory")
    parser.add_argument("-o", "--output", required=True, help="Pipeline output directory")
    parser.add_argument("-t", "--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--mode", type=str, choices=["mock", "prod"], default="prod", help="Execution mode")
    
    # New argument to catch the exact sample ID routed by step0
    parser.add_argument("--sample", type=str, required=True, help="Explicit Sample ID to process")
    parser.add_argument("--assay", type=str, required=False, default="cfdna", help="Target genetic material") # <--- Add this
    
    args = parser.parse_args()
    
    run_trimming(args.input, args.output, args.threads, args.mode, args.sample, args.assay)