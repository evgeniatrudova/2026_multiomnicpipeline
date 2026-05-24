import os
import sys
import argparse
import subprocess
import shutil

def check_gatk_env():
    if shutil.which("gatk") is None:
        print("CRITICAL SERVER ERROR: 'gatk' command utility missing from active environment.")
        sys.exit(1)

def run_variant_calling(input_dir, output_dir, threads, mode, genome_fasta, germline_vcf, sample, assay):
    print(f"--- Running Step 8: Multi-Omics Variant Engine for {sample} [Mode: {mode.upper()}, Assay: {assay.upper()}] ---")
    
    consensus_dir = os.path.join(output_dir, "5_consensus_results")
    step_out_dir = os.path.join(output_dir, "8_variant_results")
    os.makedirs(step_out_dir, exist_ok=True)

    input_bam = os.path.join(consensus_dir, f"{sample}.consensus.bam")
    if not os.path.exists(input_bam):
        print(f"ERROR: Upstream molecular consensus BAM file not found for {sample} at: {input_bam}")
        sys.exit(1)
        
    raw_vcf = os.path.join(step_out_dir, f"{sample}_raw_variants.vcf.gz")
    final_vcf = os.path.join(step_out_dir, f"{sample}_final_somatic_variants.vcf.gz")
    assay_lower = assay.lower()

    if mode == "mock":
        print(f"  > [SIMULATION] Initiating somatic likelihood profiles for: {sample}")
        with open(raw_vcf, "w") as f:
            f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nMOCK_VCF_RECORD_FOR_{sample}")
        with open(final_vcf, "w") as f:
            f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nMOCK_VCF_RECORD_FOR_{sample}")
        return

    check_gatk_env()
    
    # 1. miRNA Bypass Logic
    if assay_lower == "mirna":
        print(f"  > [PRODUCTION] miRNA detected. Bypassing Mutect2 SNV calling.")
        # Output empty valid VCF to keep downstream pipeline steps from crashing
        with open(final_vcf, "w") as f:
             f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        return

    # 2. mRNA Splice-Junction Slicing
    active_bam = input_bam
    if assay_lower in ["mrna", "rna"]:
        print(f"  > [PRODUCTION] RNA detected. Launching GATK SplitNCigarReads to mask intronic splicing...")
        split_bam = os.path.join(step_out_dir, f"{sample}_split.bam")
        split_cmd = [
            "gatk", "SplitNCigarReads",
            "-R", genome_fasta,
            "-I", input_bam,
            "-O", split_bam
        ]
        subprocess.run(split_cmd, check=True)
        active_bam = split_bam  # Hand off the sliced BAM to Mutect2

    # 3. Standard Variant Calling
    mutect_cmd = [
        "gatk", "Mutect2",
        "-R", genome_fasta,
        "-I", active_bam,
        "-O", raw_vcf,
        "--germline-resource", germline_vcf,
        "--f1r2-tar-gz", f"{raw_vcf}.f1r2.tar.gz"
    ]
    
    filter_cmd = [
        "gatk", "FilterMutectCalls",
        "-R", genome_fasta,
        "-V", raw_vcf,
        "-O", final_vcf
    ]
    
    print(f"  > [PRODUCTION] Launching active GATK Bayesian engine for: {sample}")
    try:
        subprocess.run(mutect_cmd, check=True)
        subprocess.run(filter_cmd, check=True)
        
        # Cleanup the massive intermediate split BAM if we generated one for mRNA
        if assay_lower in ["mrna", "rna"] and os.path.exists(active_bam):
            os.remove(active_bam)
            
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL SYSTEM ERROR: GATK engine pipeline threw an exception on {sample}: {e}")
        sys.exit(1)

    print(f"\nStep 8 Complete. Somatic variant call maps written to: {step_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8: Somatic Mutation Variant Engine Controller")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["mock", "prod"], default="prod")
    parser.add_argument("--genome-fasta", type=str, required=True)
    parser.add_argument("--germline-vcf", type=str, required=True)
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--assay", type=str, required=False, default="cfdna")
    
    args = parser.parse_args()
    run_variant_calling(args.input, args.output, args.threads, args.mode, args.genome_fasta, args.germline_vcf, args.sample, args.assay)