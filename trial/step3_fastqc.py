import subprocess
import os
import shutil
import sys
import argparse

def check_fastqc_installed(mode):
    # Only enforce the installation check if we are running in true production mode
    if mode == "prod" and shutil.which("fastqc") is None:
        print("ERROR: 'fastqc' not found in this environment. Is it installed in your Conda env?")
        sys.exit(1)

def run_fastqc(input_dir, output_dir, threads, mode, sample, assay):
    """Processes a single explicit sample (both R1 and R2) for quality control."""
    print(f"--- Running Step 3: FastQC Quality Check for {sample} [Mode: {mode.upper()}] ---")
    
    fastqc_out = os.path.join(output_dir, "3_fastqc_results")
    os.makedirs(fastqc_out, exist_ok=True)
    
    # Snakemake-ready: Explicitly define the expected files for this specific sample
    r1_file = os.path.join(input_dir, f"{sample}_R1_001.fastq.gz")
    r2_file = os.path.join(input_dir, f"{sample}_R2_001.fastq.gz")
    
    files_to_check = []
    if os.path.exists(r1_file):
        files_to_check.append(r1_file)
    if os.path.exists(r2_file):
        files_to_check.append(r2_file)

    if not files_to_check:
        print(f"ERROR: Missing input reads for {sample}. Expected them at: {input_dir}")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # MOCK / SIMULATION MODE
    # ----------------------------------------------------------------------
    if mode == "mock":
        print(f"  > [SIMULATION] Generating mock FastQC reports for {sample}")
        
        for file_path in files_to_check:
            base_name = os.path.basename(file_path).replace(".fastq.gz", "").replace(".fq.gz", "")
            html_mock = os.path.join(fastqc_out, f"{base_name}_fastqc.html")
            zip_mock = os.path.join(fastqc_out, f"{base_name}_fastqc.zip")
            
            for mock_file in [html_mock, zip_mock]:
                with open(mock_file, "w") as f:
                    f.write(f"MOCK FASTQC QUALITY DATA FOR {sample} ({assay.upper()})")
                    
        print(f"Step 3 Complete. Mock FastQC reports saved to: {fastqc_out}")
        return

    # ----------------------------------------------------------------------
    # PRODUCTION MODE
    # ----------------------------------------------------------------------
    # We pass the threads explicitly to FastQC's internal multithreading engine (-t)
    command = ["fastqc", "-t", str(threads), "-o", fastqc_out, "--noextract"] + files_to_check
    
    try:
        print(f"  > [PRODUCTION] Running FastQC on {len(files_to_check)} files for {sample}...")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Step 3 Complete. FastQC reports saved to: {fastqc_out}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: FastQC failed on {sample}\n{e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: FastQC Quality Check")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["mock", "prod"], default="prod", help="Execution mode")
    
    # Snakemake-ready arguments passed dynamically from step0
    parser.add_argument("--sample", type=str, required=True, help="Explicit Sample ID to process")
    parser.add_argument("--assay", type=str, required=False, default="cfdna", help="Target genetic material")
    
    args = parser.parse_args()

    # Pass the mode to the installation checker
    check_fastqc_installed(args.mode)
    
    run_fastqc(args.input, args.output, args.threads, args.mode, args.sample, args.assay)