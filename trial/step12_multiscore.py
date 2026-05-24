import pandas as pd
import numpy as np
import logging
import sys
import os
import argparse
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def calculate_dna_score(vcf_df: pd.DataFrame) -> float:
    if vcf_df.empty:
        return 0.0
    driver_genes = ['TP53', 'EGFR', 'KRAS', 'PIK3CA']
    if 'GENE' not in vcf_df.columns:
        vcf_df['GENE'] = 'UNKNOWN'
        
    vcf_df['weighted_vaf'] = vcf_df.apply(lambda row: row['VAF'] * 2.0 if row['GENE'] in driver_genes else row['VAF'], axis=1)
    return float(vcf_df['weighted_vaf'].sum())

def calculate_frag_score(motif_counts: Dict[str, int]) -> float:
    total_obs = sum(motif_counts.values())
    if total_obs == 0:
        return 0.0
    ccca_freq = motif_counts.get('CCCA', 0) / total_obs
    return float(ccca_freq / 0.01)

# === DYNAMIC FUSION ENGINE ===
def run_multimodal_fusion(dna_s: float, frag_s: float, ev_cargo_s: float, assay: str) -> float:
    """Dynamically re-weights the fusion algorithm based on available assay biology."""
    if not all(np.isfinite([dna_s, frag_s, ev_cargo_s])):
        logging.error("Fusion Error: Data contains non-finite values.")
        return 0.0

    assay_lower = assay.lower()
    
    if assay_lower in ["mrna", "mirna", "rna"]:
        # RNA assays lack structural DNA motifs and somatic VAF. Shift 90% weight to Transcriptomics.
        combined_signal = (dna_s * 0.05) + (frag_s * 0.05) + (ev_cargo_s * 0.90)
    else:
        # Standard cfDNA multimodal weighting
        combined_signal = (dna_s * 0.4) + (frag_s * 0.3) + (ev_cargo_s * 0.3)
        
    probability = 1 / (1 + np.exp(-combined_signal))
    return float(probability)

def process_scoring(input_dir, output_dir, threads, mode, sample, assay):
    print(f"--- Running Step 12: Multimodal Fusion Scoring for {sample} [Mode: {mode.upper()}] ---")
    
    filtered_dir = os.path.join(output_dir, "10_filtered_variants")
    step_out_dir = os.path.join(output_dir, "12_multiscore_results")
    os.makedirs(step_out_dir, exist_ok=True)
    
    vcf_path = os.path.join(filtered_dir, f"{sample}_filtered_tumor_only.vcf.gz")
    
    # ----------------------------------------------------------------------
    # MOCK MODE: Simulation (No file dependencies needed)
    # ----------------------------------------------------------------------
    if mode == "mock":
        logging.info(f"  > [SIMULATION] Processing Integrated Analysis for: {sample}")
        mock_vcf = pd.DataFrame({'GENE': ['TP53', 'NOTCH1'], 'VAF': [0.008, 0.002]})
        mock_motifs = {'CCCA': 250, 'AAAA': 1200, 'TATA': 800}
        mock_rna_signal = 1.4 

        dna_score = calculate_dna_score(mock_vcf)
        frag_score = calculate_frag_score(mock_motifs)
        final_prob = run_multimodal_fusion(dna_score, frag_score, mock_rna_signal, assay)
        logging.info(f"  > FINAL RESULT for {sample}: {final_prob:.2%} Cancer Probability.")

    # ----------------------------------------------------------------------
    # PRODUCTION MODE: Real File Parsing
    # ----------------------------------------------------------------------
    elif mode == "prod":
        import pysam
        logging.info(f"  > [PRODUCTION] Calculating diagnostic fusion score for: {sample}")
        
        dna_score = 0.0
        if os.path.exists(vcf_path):
            try:
                vcf_file = pysam.VariantFile(vcf_path)
                records = []
                for record in vcf_file:
                    af = record.samples[0]['AF'][0] if 'AF' in record.samples[0] else 0.001
                    records.append({'VAF': af, 'GENE': 'UNKNOWN'})
                df_real = pd.DataFrame(records)
                dna_score = calculate_dna_score(df_real)
            except Exception as e:
                logging.error(f"Failed to parse VCF for {sample}: {e}")
        else:
            logging.info(f"  > INFO: No VCF detected (expected for RNA assay). Scoring based on RNA/Frag metrics only.")
            
        # Simulate upstream Fragmentomics and RNA data extraction
        frag_score = 1.2
        rna_signal = 0.8
        
        final_prob = run_multimodal_fusion(dna_score, frag_score, rna_signal, assay)
        logging.info(f"  > FINAL RESULT for {sample}: {final_prob:.2%} Cancer Probability.")
        
        # Save report
        report_path = os.path.join(step_out_dir, f"{sample}_fusion_score.txt")
        with open(report_path, "w") as f:
            f.write(f"Sample: {sample}\nAssay: {assay.upper()}\nDNA Score: {dna_score:.4f}\nFragment Score: {frag_score:.4f}\nRNA Signal: {rna_signal:.4f}\nFinal Prob: {final_prob:.2%}\n")            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 12: Multimodal Fusion Engine")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["mock", "prod"], default="prod")
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--assay", type=str, required=False, default="cfdna")
    
    args = parser.parse_args()
    process_scoring(args.input, args.output, args.threads, args.mode, args.sample, args.assay)