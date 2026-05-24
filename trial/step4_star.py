import os
import sys
import argparse
import subprocess
import shutil

def check_star_env(genome_dir):
    if shutil.which("STAR") is None:
        print("CRITICAL SERVER ERROR: 'STAR' command-line utility not found.")
        sys.exit(1)
    if not os.path.exists(os.path.join(genome_dir, "SAindex")):
        print(f"CRITICAL SERVER ERROR: No valid STAR index at: {genome_dir}")
        sys.exit(1)

def run_star_alignment(input_dir, output_dir, threads, mode, star_index, sample, assay):
    print(f"--- Running Step 4: STAR Genomic Alignment for {sample} [Mode: {mode.upper()}, Assay: {assay.upper()}] ---")
    
    trimmed_dir = os.path.join(output_dir, "1_trimmed_reads")
    step_out_dir = os.path.join(output_dir, "4_alignment_results")
    os.makedirs(step_out_dir, exist_ok=True)

    r1_file = os.path.join(trimmed_dir, f"{sample}_R1_trimmed.fastq.gz")
    r2_file = os.path.join(trimmed_dir, f"{sample}_R2_trimmed.fastq.gz")

    if not os.path.exists(r1_file) or not os.path.exists(r2_file):
        print(f"ERROR: Missing trimmed fastq.gz reads for {sample} in path: {trimmed_dir}")
        sys.exit(1)

    output_prefix = os.path.join(step_out_dir, f"{sample}_")
    
    if mode == "mock":
        print(f"  > [SIMULATION] Generating mock coordinate-sorted alignment file for {assay.upper()}")
        with open(f"{output_prefix}Aligned.sortedByCoord.out.bam", "w") as f:
            f.write(f"MOCK GENOMIC BAM ALIGNMENT DATA FOR {sample}")
        return
            
    check_star_env(star_index)
    
    # 1. Core Parameters (Universal)
    base_star_cmd = [
        "STAR",
        "--runThreadN", str(threads),
        "--genomeDir", star_index,
        "--readFilesIn", r1_file, r2_file,
        "--readFilesCommand", "zcat",
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outFileNamePrefix", output_prefix,
        "--outSAMattributes", "NH", "HI", "NM", "MD"
    ]

    # 2. Dynamic Biological Routing
    assay_lower = assay.lower()
    if assay_lower in ["mrna", "rna"]:
        print("  > [ASSAY CONFIG] RNA detected. Enabling basic splice-junction mapping and GeneCounts.")
        assay_params = [
            "--twopassMode", "Basic",
            "--quantMode", "TranscriptomeSAM", "GeneCounts",
            "--alignSJoverhangMin", "8"
        ]
    elif assay_lower == "mirna":
        print("  > [ASSAY CONFIG] miRNA detected. Locking introns and minimizing match filters.")
        assay_params = [
            "--alignIntronMax", "1",
            "--outFilterMatchNmin", "16",
            "--outFilterMismatchNmax", "1"
        ]
    else:
        print("  > [ASSAY CONFIG] cfDNA detected. Enforcing continuous genomic mapping (No Splicing).")
        assay_params = [
            "--alignIntronMax", "1",
            "--outFilterMultimapNmax", "20"
        ]

    star_command = base_star_cmd + assay_params
    
    print(f"  > [PRODUCTION] Launching active algorithmic alignment for: {sample}")
    try:
        subprocess.run(star_command, check=True)
        print(f"SUCCESS: Mapped BAM written to {output_prefix}Aligned.sortedByCoord.out.bam")
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL SYSTEM ERROR: STAR alignment binary crashed on sample {sample}.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: STAR Genomic Mapping")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["mock", "prod"], default="prod")
    parser.add_argument("--star-index", type=str, required=False)
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--assay", type=str, required=False, default="cfdna")
    
    args = parser.parse_args()
    run_star_alignment(args.input, args.output, args.threads, args.mode, args.star_index, args.sample, args.assay)