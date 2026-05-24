import os
import sys
import argparse
import subprocess

# 1. THE TRUTH TABLE
PIPELINE_MAP = {
    "independent": {
        "cfdna": [
            (1, "step1_cutadapt.py", "Trim"), (2, "step2_umis.py", "UMI"), (3, "step3_fastqc.py", "QC"),
            (4, "step4_star.py", "Map"), (5, "step5_fgbio.py", "Dedupe"), (6, "step6_samtools.py", "Fragment"),
            (7, "step7_fragmentomics.py", "Motif"), (8, "step8_variant.py", "Variant"), (9, "step9_variant_vis.py", "VAF-Plot"),
            (10, "step10_chip.py", "Noise-Reduct"), (11, "step11_maftools.py", "Viz"), (12, "step12_multiscore.py", "Fusion"),
            (13, "step13_clinical.py", "Clinical"), (14, "step14_dynamic.py", "Evolution")
        ],
        "mrna": [
            (1, "step1_cutadapt.py", "Trim"), (2, "step2_umis.py", "UMI"), (3, "step3_fastqc.py", "QC"),
            (4, "step4_star.py", "Map"), (5, "step5_fgbio.py", "Dedupe"), (6, "step6_samtools.py", "Fragment"),
            (7, "step7_fragmentomics.py", "Motif"), (8, "step8_variant.py", "Variant"),
            (10, "step10_chip.py", "Noise-Reduct"), (11, "step11_maftools.py", "Viz"), (12, "step12_multiscore.py", "Fusion"),
            (13, "step13_clinical.py", "Clinical"), (14, "step14_dynamic.py", "Evolution")
        ],
        "mirna": [
            (1, "step1_cutadapt.py", "Trim"), (2, "step2_umis.py", "UMI"), (3, "step3_fastqc.py", "QC"),
            (4, "step4_star.py", "Map"), (5, "step5_fgbio.py", "Dedupe"), (6, "step6_samtools.py", "Fragment"),
            (7, "step7_fragmentomics.py", "Motif"), 
            (10, "step10_chip.py", "Noise-Reduct"), (11, "step11_maftools.py", "Viz"),
            (13, "step13_clinical.py", "Clinical"), (14, "step14_dynamic.py", "Evolution")
        ]
    }
}

# --- HARDCODED SERVER PATHS (HIDDEN FROM STUDENTS) ---
GENOME_FASTA = "/disk2/radgro/projects/rna_seq_mapping/genome/homo_sapiens/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
GERMLINE_VCF = "/disk2/radgro/projects/rna_seq_mapping/genome/homo_sapiens/gnomad.genomes.v3.1.2.sites.chr17.vcf.bgz"

def get_blueprint(assay):
    return PIPELINE_MAP["independent"][assay]

def run_step(sample, script_name, script_path, input_dir, output_dir, threads, mode, assay):
    # Passes the data invisibly to the sub-scripts
    cmd = ["python", script_path, "-i", input_dir, "-o", output_dir, "-t", str(threads), "--mode", mode, "--sample", sample, "--assay", assay]
    
    if script_name == "step8_variant.py":
        cmd.extend(["--genome-fasta", GENOME_FASTA, "--germline-vcf", GERMLINE_VCF])
    
    print(f"  ▶️ Running {script_name}...")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student-Safe Orchestrator")
    parser.add_argument("--raw_data", required=True, help="Link to the folder containing raw fastq files")
    parser.add_argument("--sample", required=True, help="Name of the sample")
    parser.add_argument("--assay", choices=["cfdna", "mrna", "mirna"], required=True)
    parser.add_argument("--mode", choices=["mock", "prod"], default="prod")
    parser.add_argument("-t", "--threads", type=int, default=16)
    parser.add_argument("--start-at", type=int, default=1, help="Step number to resume from if pipeline crashed")
    args = parser.parse_args()

    # --- AUTOMATIC FOLDER GENERATION ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    
    # 1. Create the Master Output Directory based on the Mode (mock_results or prod_results)
    master_mode_folder = f"{args.mode}_results"
    master_mode_path = os.path.join(ROOT_DIR, master_mode_folder)
    os.makedirs(master_mode_path, exist_ok=True)
    
    # 2. Create the specific student result folder inside that master directory
    result_folder_name = f"RESULTS_{args.sample}_{args.assay.upper()}"
    final_output_dir = os.path.join(master_mode_path, result_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"\n📂 DATA ROUTING:")
    print(f"  ▶ Pulling Raw Data From: {args.raw_data}")
    print(f"  ▶ Pushing Results To:    {final_output_dir}")

    active_steps = get_blueprint(args.assay)
    
    for step_number, script_name, step_desc in active_steps:
        # Crash Recovery Logic: Skip steps if start-at is greater than current step
        if step_number < args.start_at:
            print(f"  ⏭️ BYPASSING Step {step_number}: {step_desc} (Resuming at {args.start_at})")
            continue
            
        script_path = os.path.join(SCRIPT_DIR, script_name)
        run_step(args.sample, script_name, script_path, args.raw_data, final_output_dir, args.threads, args.mode, args.assay)