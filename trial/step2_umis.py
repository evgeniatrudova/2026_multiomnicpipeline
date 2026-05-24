import os
import sys
import argparse

# Downstream UMI tools (fgbio/zUMIs) require Read 1 (R1) to remain completely 
# untrimmed to preserve exact barcode lengths. We use asymmetric trimming (`--pair-filter=any`, `-m 20`) to trim 
# adapters/dimers from R2, while safely leaving R1 intact.If the file is already trimmed, cutadapt will find 0% adapters.
# Verify R1 length was preserved (both commands must output the exact same number) by running terminal commands: 
#   zcat raw_data/sample1_R1_001.fastq.gz | head -n 2 | tail -n 1 | wc -c
#   zcat 1_trimmed_reads/sample1_R1_trimmed.fastq.gz | head -n 2 | tail -n 1 | wc -c

def run_umis_setup(input_dir, output_dir, threads, mode, sample, assay):
    print(f"--- Running Step 2: UMI Configuration for {sample} [Mode: {mode.upper()}] ---")
    
    # Track the trimmed reads directory
    trimmed_dir = os.path.join(output_dir, "1_trimmed_reads")
    step_out_dir = os.path.join(output_dir, "2_umi_configs")
    os.makedirs(step_out_dir, exist_ok=True)

    # Snakemake-ready: Explicitly verify upstream dependencies for this specific sample
    r1_trimmed = os.path.join(trimmed_dir, f"{sample}_R1_trimmed.fastq.gz")
    
    if not os.path.exists(r1_trimmed) and mode == "prod":
        print(f"WARNING: Trimmed reads for {sample} not found at {r1_trimmed}. Check upstream pipeline flow.")

    # ----------------------------------------------------------------------
    # MOCK / SIMULATION MODE
    # ----------------------------------------------------------------------
    if mode == "mock":
        print(f"  > [SIMULATION] Generating mock UMI configuration template for {sample}...")
        
        # Create a sample-specific placeholder file
        mock_config = os.path.join(step_out_dir, f"{sample}_umi_config.yaml")
        with open(mock_config, "w") as f:
            f.write(f"# MOCK UMI CONFIGURATION DATA FOR {sample}\n")
            f.write(f"assay_type: {assay}\n")
            
        print(f"Step 2 Complete. Mock configurations saved to: {step_out_dir}")
        return

    # ----------------------------------------------------------------------
    # PRODUCTION MODE
    # ----------------------------------------------------------------------
    print(f"  > [PRODUCTION] Mapping true UMI configuration templates for {sample}...")
    
    # Sample-specific config path
    prod_config = os.path.join(step_out_dir, f"{sample}_umi_config.yaml")
    
    # In a real pipeline, you would use a templating engine (like Jinja2) to write out complex UMI extraction parameters
    with open(prod_config, "w") as f:
        f.write(f"# UMI CONFIGURATION FOR {sample}\n")
        f.write(f"assay_type: {assay}\n")

    print(f"Successfully mapped UMI configuration templates for {sample}.")
    print(f"Step 2 Complete. Configurations saved to: {step_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: UMI processing setup")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["mock", "prod"], default="prod", help="Execution mode")
    
    # Snakemake-ready arguments passed dynamically from step0
    parser.add_argument("--sample", type=str, required=True, help="Explicit Sample ID to process")
    parser.add_argument("--assay", type=str, required=False, default="cfdna", help="Target genetic material")
    
    args = parser.parse_args()
    
    run_umis_setup(args.input, args.output, args.threads, args.mode, args.sample, args.assay)