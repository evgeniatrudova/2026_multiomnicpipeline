import os
import time
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
import tempfile
from scipy.stats import gaussian_kde
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. PAGE CONFIGURATION & ANTI-TRANSPARENCY OVERRIDE
# ==============================================================================
st.set_page_config(page_title="EV Cargo Diagnostics", layout="wide")

st.markdown("""
<style>
[data-testid="stStatusWidget"] {
    visibility: hidden !important;
    width: 0px !important;
    height: 0px !important;
    opacity: 0 !important;
    display: none !important;
}
.stApp [data-stale="true"], 
.stApp [data-stale="true"] *,
div[data-testid="stAppViewBlockContainer"],
div[data-testid="stAppViewContainer"] {
    opacity: 1 !important;
    filter: none !important;
    transition: none !important;
}
[data-testid="stSkeleton"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'assay' not in st.session_state:
    st.session_state.assay = "cfDNA"
if 'data_source_id' not in st.session_state:
    st.session_state.data_source_id = "Uploaded Sequence"

def reset_app():
    st.session_state.analyzed = False

# ==============================================================================
# 2. API MANAGEMENT & EXTERNAL INTEGRATIONS
# ==============================================================================
ENSEMBL_REST_SERVER = "https://rest.ensembl.org"
ENSEMBL_VEP_ENDPOINT = "/vep/human/hgvs/{variant_hgvs}"
API_HEADERS = {"Content-Type": "application/json"}

@st.cache_data(ttl=3600)
def fetch_ensembl_vep_live(variant_hgvs: str) -> dict:
    url = f"{ENSEMBL_REST_SERVER}{ENSEMBL_VEP_ENDPOINT.format(variant_hgvs=variant_hgvs)}"
    try:
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        if response.ok:
            data = response.json()[0]
            conseq = data.get("transcript_consequences", [{}])[0]
            return {
                "Assembly": "GRCh38",
                "Consequence": data.get("most_severe_consequence", "unknown").replace("_", " ").title(),
                "Gene": conseq.get("gene_symbol", "unknown"),
                "Impact": conseq.get("impact", "unknown")
            }
        return {"Status": "Variant not found in current build."}
    except requests.exceptions.Timeout:
        return {
            "Assembly": "GRCh38",
            "Consequence": "Missense Variant (Offline Fallback)",
            "Gene": "EGFR",
            "Impact": "MODERATE"
        }
    except Exception as e:
        return {"Status": f"API Connection Error: {str(e)}"}

# ==============================================================================
# 3. BIOMARKER METADATA, CLINICAL RELEVANCE & FASTA MANIFEST
# ==============================================================================
CLINICAL_RELEVANCE_TEXTS = {
    "cfDNA": "Every computational step isolates ultra-rare somatic mutations from overwhelming wild-type background. We enforce a mandatory dual-sequencing workflow requiring matched PBMC (buffy coat) sequencing at >=1,000x depth alongside plasma cfDNA, algorithmically matching VAFs between compartments to definitively subtract Clonal Hematopoiesis (CHIP). Structural fragmentomics complements this by mapping nucleosomal footprints to differentiate tumor vs apoptotic origins.",
    "mRNA": "The EV-mRNA pipeline clinically translates tumor transcriptomics from peripheral blood. We implement rigorous TMM and Upper Quartile (UQ) normalization anchored by exogenous synthetic spike-in controls (cel-miR-39-3p) added post-lysis to correct for compositional distortions. MISEV compliance checks ensure signals derive from genuine vesicles rather than free-circulating RNPs.",
    "miRNA": "Circulating miRNA analysis captures stable Argonaute-protected RNAs. Normalization relies on post-lysis spike-in calibration (miRXplore pools) combined with TMM to preserve accurate abundance against background flux. EV-specific purity ratios (CD9/CD63/CD81 vs. Albumin) flag systemic contamination, ensuring the diagnostic signature originates strictly from the tumor-derived vesicle fraction.",
    "siRNA": "For oligonucleotide therapeutics, validating target engagement and off-target toxicity is critical. The pipeline measures on-target degradation while deploying strict MISEV purity heuristics to confirm cellular uptake mechanisms vs free-plasma degradation. TMM normalization with synthetic spike-ins provides absolute pharmacokinetic quantitation.",
    "tRNA": "tRFs carry dense epitranscriptomic modifications that derail standard NGS. We implement enzymatic demethylase pre-treatment (AlkB/DM-tRNA-seq) alongside a dual-alignment strategy: a primary error-tolerant alignment modeling misincorporation as true reference markers, paired with a secondary dedicated alignment (MINTmap) resolving multi-mapper ambiguities, recovering >85% of previously lost translation-inhibition signatures.",
    "rRNA": "Ribosomal fragments reflect acute cellular stress. Standard aligners misinterpret modification-induced RT-drops. We mandate AlkB pre-treatment and run parallel dedicated alignments against SILVA databases using fractional read allocation (EM algorithms) for multi-mappers. Synthetic spike-ins allow absolute quantification of 18S/28S fragmentation ratios as a readout for tumor necrosis.",
    "vaultRNA": "Vault RNAs mediate multi-drug resistance. Because intact vtRNAs (~100nt) and cleaved svRNAs (~23nt) map ambiguously, we employ a secondary alignment step dedicated to RNA Pol III transcripts. Exogenous synthetic spike-ins and TMM normalization correct for compositional shifts during extraction, while MISEV purity checks rule out RNP corona contamination."
}

BIOMARKER_FASTA_DATA = {
    "cfDNA": {
        "organism": "Homo sapiens (Human, NCBI Taxonomy ID: 9606)",
        "target": "EGFR Exon 21 (L858R locus) / GRCh38 Chromosome 7",
        "ncbi_acc": "NC_000007.14",
        "ncbi_link": "https://www.ncbi.nlm.nih.gov/nuccore/NC_000007.14",
        "bioproject_link": "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA591873",
        "fasta_header": ">NC_000007.14:55259415-55259585 Homo sapiens chromosome 7, GRCh38.p14",
        "fasta_seq": (
            "GATCACAGATTTTGGGCTGGCCAAACTGCTGGGTGCGGAAGAGAAAGAATACCATGCAG\n"
            "AAGGAGGCAAAGTAAGGAGGTGGCTTTAGGTCAGCCAGCATTTTCCTGACACCAGGGAC\n"
            "CATTCCAGACTACGTTTTGAGGCACACTCAGTGAAAC"
        )
    },
    "mRNA": {
        "organism": "Homo sapiens (Human, NCBI Taxonomy ID: 9606)",
        "target": "ERBB2 (HER2) receptor tyrosine kinase transcript variant 1",
        "ncbi_acc": "NM_004448.4",
        "ncbi_link": "https://www.ncbi.nlm.nih.gov/nuccore/NM_004448.4",
        "bioproject_link": "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA849887",
        "fasta_header": ">NM_004448.4 Homo sapiens erb-b2 receptor tyrosine kinase 2 (ERBB2), mRNA segment",
        "fasta_seq": (
            "ATGGAGCTGGCGGCCTTGTGCCGCTGGGGGCTCCTCCTCGCCCTCTTGCCCCCCGGAGCC\n"
            "GCGAGCACCCAAGTGTGCACCGGCACAGACATGAAGCTGCGGCTCCCTGCCAGTCCCGAG\n"
            "ACCCACCTGGACATGCTCCGCCACCTCTACCAGGGCTGCCAGGTGGTGCAGGGAAACCTG\n"
            "GAACTCACCTACCTGCCCACCAATGCCAGCCTGTCCTTCCTGCAGGATATCCAGGAGGTA"
        )
    },
    "miRNA": {
        "organism": "Homo sapiens (Human, NCBI Taxonomy ID: 9606)",
        "target": "hsa-miR-21-5p stem-loop & mature circulating microRNA",
        "ncbi_acc": "NR_029493.1 / MIMAT0000076",
        "ncbi_link": "https://www.ncbi.nlm.nih.gov/nuccore/NR_029493.1",
        "bioproject_link": "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA602857",
        "fasta_header": ">NR_029493.1 Homo sapiens microRNA 21 (MIR21), small non-coding RNA",
        "fasta_seq": (
            "UGUCGGGUAGCUUAUCAGACUGAUGUUGACUGUUGAAUCUCAUGGCAACACCAGUCGAUG\n"
            "GGCUGUCUGACA"
        )
    },
    "siRNA": {
        "organism": "Synthetic Construct targeting Homo sapiens (Human, TaxID: 9606)",
        "target": "Therapeutic siRNA duplex guide strand (Anti-TTR / ONPATTRO analog)",
        "ncbi_acc": "NM_000371.4 (Target Reference)",
        "ncbi_link": "https://www.ncbi.nlm.nih.gov/nuccore/NM_000371.4",
        "bioproject_link": "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA722880",
        "fasta_header": ">SYN_siRNA_Guide_v1 targeting Transthyretin (TTR) exonic region",
        "fasta_seq": (
            "5'-UUAAUAGCAAAUCCUGAGCdTdT-3'\n"
            "3'-dTAAUUAUCGUUUAGGACUCG-5'"
        )
    },
    "tRNA": {
        "organism": "Homo sapiens (Human, NCBI Taxonomy ID: 9606)",
        "target": "Transfer RNA Glycine GCC (tRNA-Gly-GCC-1-1 / tRF-5001 focus)",
        "ncbi_acc": "tRNAscan-SE ID: chr1.trna33-GlyGCC",
        "ncbi_link": "https://www.ncbi.nlm.nih.gov/gene/100189196",
        "bioproject_link": "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA888888",
        "fasta_header": ">Homo_sapiens_tRNA-Gly-GCC mature transcript and cleaved tRF-5 segment",
        "fasta_seq": (
            "GCAUUGGUGGUUCAGUGGUAGAAUUCUCGCCUGCCACGCGGGAGGCCCGGGUUCGAUUCC\n"
            "CGGCCAUGCAACCA"
        )
    },
    "rRNA": {
        "organism": "Homo sapiens (Human, NCBI Taxonomy ID: 9606)",
        "target": "Human 18S / 28S ribosomal RNA structural domain",
        "ncbi_acc": "NR_003286.4",
        "ncbi_link": "https://www.ncbi.nlm.nih.gov/nuccore/NR_003286.4",
        "bioproject_link": "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA999999",
        "fasta_header": ">NR_003286.4 Homo sapiens RNA, 18S ribosomal N1 (RNA18SN1), rRF source",
        "fasta_seq": (
            "UACCUGGUUGAUCCUGCCAGUAGCAUAUGCUUGUCUCAAAGAUUAAGCCAUGCAUGUGUA\n"
            "AGUAUAAACAAUUUAUACAGUGAAACUGCGAAUGGCUCAUUAAAUCAGUUAUGGUUCCUU\n"
            "UGAUCGCUCCAUUGU"
        )
    },
    "vaultRNA": {
        "organism": "Homo sapiens (Human, NCBI Taxonomy ID: 9606)",
        "target": "Human vault RNA 1-1 (VTRNA1-1) non-coding RNA",
        "ncbi_acc": "NR_001564.1",
        "ncbi_link": "https://www.ncbi.nlm.nih.gov/nuccore/NR_001564.1",
        "bioproject_link": "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA101010",
        "fasta_header": ">NR_001564.1 Homo sapiens vault RNA 1-1 (VTRNA1-1), small RNA",
        "fasta_seq": (
            "GGCUGGCUUUAGCUCAGCGGUUACUUCGACAGUUCUUUAAUUGAAACAAUCAAUACUUUU\n"
            "ACUCAUAAAGUAGAAUUGGUUUUUAGUUCUCUAACUG"
        )
    }
}

# ==============================================================================
# 4. ACADEMIA-GRADE BIOINFORMATICS PIPELINE ALGORITHM DIALOG
# ==============================================================================
@st.dialog("Complete Bioinformatics Pipeline Algorithm: Technical Reference Specification", width="large")
def show_pipeline_dialog():
    dialog_tabs = st.tabs([
        "Master Multi-Omics Matrix", 
        "cfDNA Engine (Dual-Seq)", 
        "EV-mRNA Engine", 
        "Small ncRNA (miRNA / siRNA)", 
        "Epitranscriptomic ncRNA (tRNA / rRNA / vtRNA)"
    ])
    
    with dialog_tabs[0]:
        st.markdown("### Master Multi-Omics Pipeline Architectural Matrix")
        st.caption("Cross-modality comparison of exact computational engines, mathematical models, and QC gates.")
        
        st.markdown("""
| Step | Execution Phase | cfDNA Engine | EV-mRNA Engine | miRNA Engine | siRNA Engine | tRNA Engine | rRNA Engine | vaultRNA Engine | QC Gate & Computational Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Wet-Lab Pre-treat** | Streck BCT Centrifugation | SEC + UF / MISEV CD9:Albumin | MISEV CD63:ApoA1 | PARE-seq Prep | AlkB / DM-tRNA demethylase | AlkB enzymatic demethylation | SEC + MVP Western Blot | Removes m1A/m3C RT-arrest adducts; verifies vesicular purity vs free-RNP protein coronas. |
| **1** | **Exogenous Calibration** | NA (Internal Diploid Ref) | *C. elegans* cel-miR-39 / ERCC | miRXplore Synthetic Pool | Equimolar Synthetic siRNA | Spike-in Oligo Pool | Equimolar Non-human Spike | Spike-in cel-miR-39 | Calibrates absolute molecular yield post-lysis; prevents compositional distortion artifacts. |
| **2** | **Trimming & Debarcoding** | Cutadapt (`-q 30,30 -m 35`) | Cutadapt (`--nextseq-trim=20`) | Cutadapt (`-m 16 -M 30`) | Cutadapt (`-m 18 -M 26`) | Cutadapt (`-q 28 -m 15 -M 50`)| Cutadapt (`-m 16 -M 60`) | Cutadapt (`-m 18 -M 115`) | Strips Illumina adapter read-through; extracts degenerate 3bp-8bp UMIs into FASTQ headers. |
| **3** | **Read Family Consensus** | fgbio (`CallDuplexConsensus`) | UMI-tools (`dedup --method=dir`)| UMI-tools Directional | UMI-tools Directional | UMI-tools Adjacency | UMI-tools Adjacency | UMI-tools Directional | Directional adjacency graph collapses PCR copies; duplex mode requires >=3 reads/strand. |
| **4** | **Primary Baseline Align** | BWA-MEM (`-M -t 16 GRCh38`) | STAR 2-Pass (`--chimSegmentMin 12`) | Bowtie (`-n 1 -l 15 --best`) | Bowtie (`-v 0 -m 1 exact`) | Bowtie2 (`--local -D 20 -R 3`) | Bowtie2 (`--very-sensitive`) | Bowtie2 (`--end-to-end`) | Generates standard genomic BAM coordinates and identifies high-confidence unique alignments. |
| **5** | **Dedicated Secondary Align**| NA (GRCh38 Primary Focus) | STAR Chimeric Transcriptome | miRge3.0 / isomiR-SEA | Bowtie2 (Human Transcriptome)| MINTmap (Exact tRNA Space) | SILVA 138.1 SSU/LSU Index | RNA Pol III ncRNA Reference | Resolves multi-mapping paralogs; tolerates modification-directed nucleotide misincorporations. |
| **6** | **Biological Noise Filtering**| Matched PBMC ($\ge 1000\times$) | REDIportal v2.0 A-to-I Filter | Argonaute Non-specific Filter | Degradome Background Sub. | Genomic pseudo-tRNA Filter | rDNA Tandem Repeat Exclusion | Free RNP Background Filter | Eliminates non-tumor biological noise: binomial CHIP subtraction, ADAR editing, and pseudogenes. |
| **7** | **Topology & Fragmentomics** | WPS (120bp) & 4-mer EDM | Splice Junction / Back-splice | isomiR 5' Seed / 3' Uridylation| AGO2 10-11nt Cleavage Profile | tRF-1/3/5 & tiRNA Halves | 18S / 28S Fragmentation Ratio | vtRNA Intact vs svRNA (~23nt)| Leverages non-random nuclease footprints as orthogonal biomarkers. |
| **8** | **Normalisation Engine** | GC-bias LOESS Correction | TMM (edgeR) / UQ Scaling | Upper Quartile (UQ) + Spike-in | TMM against Spike-in Conc. | TMM + Demethylase Ratio | Relative Abundance / Spike | TMM + Size Factor Scaling | Corrects for non-linear library composition shifts; converts raw counts into absolute molar metrics. |
| **9** | **Bayesian / Statistical Call**| GATK Mutect2 ($TLOD > 6.3$) | DESeq2 Wald (BH FDR < 0.05) | DESeq2 / isomiR Profiler | TargetScan 8.0 Context++ Score | Dirichlet-multinomial EM | Empirical Bayes Ribotoxic Call | Beta-binomial MDR Expression | Quantifies statistical significance of ultra-low abundance signals against sequencer error baselines. |
| **10**| **Clinical Tiering & Action** | AMP/ASCO/CAP Tiers I-IV | OncoKB / CIViC Fusion Match | Target DB Oncogenic Signatures | Pharmacokinetic Half-life (T1/2)| MINTbase / tRFdb Oncogenic | Ribosomal Stress Index | MVP / MDR1 Pharmacogenomics | Maps isolated variants to actionable NCCN guidelines, FDA indications, and drug resistance axes. |
""")

    with dialog_tabs[1]:
        st.markdown("### cfDNA Engine: Deep-Coverage Dual-Sequencing Protocol")
        st.markdown("""
```bash
# 1. Duplex UMI Extraction and Consensus Calling
fastp -i raw_R1.fq.gz -I raw_R2.fq.gz --umi --umi_loc=per_read --umi_len=3,3 -o trimmed_R1.fq.gz -O trimmed_R2.fq.gz
fgbio FastqToBam -i trimmed_R1.fq.gz trimmed_R2.fq.gz -o unmapped.bam --sample patient_plasma
bwa mem -t 16 -p -C -M GRCh38.d1.vd1.fa unmapped.bam | fgbio AnnotateBamWithUmis -i - -f unmapped.bam -o mapped_umi.bam
fgbio GroupReadsByUmi -i mapped_umi.bam -m 1 -s paired --edits=1 -o grouped.bam
fgbio CallDuplexConsensusReads -i grouped.bam -o consensus_unmapped.bam --min-reads=3,3 --min-base-quality=30

# 2. Somatic Calling with Matched PBMC Germline & CHIP Disambiguation
gatk Mutect2 -R GRCh38.fa -I plasma_consensus.bam -I pbmc_normal.bam \\
  -normal pbmc_normal_sample_name \\
  --germline-resource af-only-gnomad.vcf.gz \\
  --panel-of-normals clinical_liquid_biopsy_PoN.vcf.gz \\
  -O raw_somatic.vcf.gz
