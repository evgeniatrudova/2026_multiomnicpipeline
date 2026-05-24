Multi-Omics Liquid Biopsy Pipeline
This automated pipeline is designed for the detection of ultra-low frequency somatic mutations. It dynamically routes data and selects appropriate analytical steps based on your target assay: cfDNA, mRNA, or miRNA.

To optimize computational resources and ensure system stability, this pipeline operates on a strict two-tier architecture:

Mock Mode: A lightweight, local simulation to verify structural integrity and connectivity.

Production Mode: Heavy, server-side execution for biological data analysis.

⚠️ WARNING: You must never run a Production execution without first passing a Mock test to validate your environment.

Step 1: Environment & Dependencies
Before running the pipeline, activate your Conda environment and install the required bioinformatic tools:

Bash
conda install -c bioconda -c conda-forge cutadapt fastqc star fgbio samtools gatk4 ensembl-vep pyyaml pysam pandas numpy matplotlib seaborn -y
Troubleshooting Missing Dependencies:
If the pipeline fails, verify the tool's installation and reinstall it using these commands (replace [dependency_name] with the missing tool):

Bash
conda --version
which [dependency_name]   
conda install -c bioconda [dependency_name]
Step 2: Directory & File Preparation
Download the Pipeline: Place the 2026_multiomnicpipeline directory in an accessible location (e.g., your Desktop).

Windows Users: To find your exact path, right-click the folder and select "Copy as path".

Reference Genome: Ensure the GRCh38 Human Reference Genome is downloaded to the server. If it is hosted centrally, obtain the direct directory path to its location.

Step 3: Server Connectivity
Log into your institutional server to locate your raw FASTQ data and the pipeline scripts:

Bash
# Connect to the server
ssh student@uni.edu

# Verify your current directory
pwd
ls

# Navigate to the pipeline trial directory
cd /path/to/2026_multiomnicpipeline/trial

# Verify your raw data files are present
ls -lh /path/to/raw_data/
Step 4: Local Testing (Mock Mode)
Always validate your setup locally before initiating a server run. Ensure your terminal is inside the 2026_multiomnicpipeline/trial directory. This command pulls dummy data from the mock folder and generates a test result folder:

Bash
python step0_main_pipeline.py \
  --raw_data "/path/to/2026_multiomnicpipeline/mock" \
  --sample MIRNA_TEST_01 \
  --assay mirna \
  -t 4 \
  --mode mock
Step 5: Server Execution (Production Mode)
Once the mock test passes, you are ready to analyze real biological data.

Command-Line Arguments:

--raw_data: Absolute path to the folder containing your raw .fastq.gz files.

--sample: Designated name for your sample (e.g., TUMOR_01).

-t: CPU threads to allocate (default 16; lower to 8 or 4 if the server is under heavy load).

--start-at: Crash recovery; change to the step number where the pipeline stopped to resume.

--mode prod: Flags the pipeline for full biological analysis.

Select the command corresponding to your assay:

cfDNA

Bash
python step0_main_pipeline.py \
  --raw_data /path/to/my_raw_data_folder \
  --sample TUMOR_CFDNA_01 \
  --assay cfdna \
  -t 16 \
  --start-at 1 \
  --mode prod

## mRNA
Bash
python step0_main_pipeline.py \
  --raw_data /path/to/my_raw_data_folder \
  --sample TUMOR_MRNA_01 \
  --assay mrna \
  -t 16 \
  --start-at 1 \
  --mode prod


## miRNA
Bash
python step0_main_pipeline.py \
  --raw_data /path/to/my_raw_data_folder \
  --sample TUMOR_MIRNA_01 \
  --assay mirna \
  -t 16 \
  --start-at 1 \
  --mode prod
