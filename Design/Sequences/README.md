# Sequences Directory

This directory contains the workflow for converting CIPHER encoding weights into fully assembled probe sequences ready for experimental use. The pipeline transforms abstract bit assignments from the CIPHER model into complete DNA probe sequences with readout arms, primers, and restriction sites.

## Overview

The Sequences directory implements a multi-step pipeline that:
1. Generates optimized 30-mer readout arm sequences (currently used to map bits to genes)
2. Generates optimized 20-mer sequences for future bDNA amplification applications (not currently used)
3. Converts CIPHER encoding weights to specific probe selections
4. Assembles complete probe sequences with all required components
5. Outputs a final `Probes.csv` file ready for synthesis

## Directory Contents

### Notebooks

#### 1. `super_probe_30mer_arms_2025Apr24.ipynb`
**Purpose**: Generates optimized 30-mer readout arm sequences used to map bits to genes in the current probe assembly workflow.

**Key Functions**:
- `generate_base_sequences()`: Generates candidate sequences with specified constraints (length, melting temperature, GC content)
- `nupack_wrapper()`: Calculates minimum free energy (MFE) for secondary structure analysis
- `fast_match_count_unfold()`: Efficiently searches for homologous sequences in transcriptome
- `calculate_mfe_pair()`: Evaluates dimer formation and off-target binding using NUPACK

**Process**:
1. Generates candidate sequences meeting basic criteria (GC content 25-75%, melting temperature within ±5°C of target)
2. Filters sequences to avoid restriction sites (e.g., NotI-HF: `GCGGCCGC`)
3. Evaluates each sequence for:
   - Secondary structure formation (self-folding)
   - Dimer formation with reverse complement
   - Off-target binding to transcriptome
4. Selects sequences with favorable properties (low secondary structure, minimal dimer formation, minimal off-target binding)
5. Saves optimized sequences to `optimized_sequences_30mers.csv`

**Output**: `optimized_sequences_30mers.csv` - Contains ID, Sequence, Arm1, and various energy scores (secondary, dimer, off-target)

**Current Usage**: These 30-mer sequences are used as readout arms in `make_fasta.ipynb` to map bits to genes. Each bit is assigned a pair of unique 30-mer arms (arm_0 and arm_1) that enable bit-specific detection.

#### 2. `super_probe_20mer_arms_2025Apr30.ipynb`
**Purpose**: Generates optimized 20-mer sequences designed for future use as bDNA (branched DNA) amplification sequences.

**Note**: These sequences are **not currently used** in the probe assembly workflow. They are prepared for future experimental applications that will utilize bDNA amplification technology.

**Key Functions**: Same as 30mer notebook (see above)

**Process**: Same workflow as 30mer version but generates 20-mer sequences

**Output**: `optimized_sequences_20mers.csv` - Contains optimized 20-mer sequences with same structure as 30mers

#### 3. `make_fasta.ipynb`
**Purpose**: Main assembly notebook that converts CIPHER encoding weights into complete probe sequences.

**Key Functions**:
- `filter_csv_to_df()`: Filters encoding site database to genes used in CIPHER design
- `optimize_weight_matrix_to_fasta()`: Converts encoding weights to specific probe selections
- `revcomp()`: Computes reverse complement of DNA sequences
- `fuzzy_search()`: Searches for restriction sites with error tolerance
- `assemble_probe()`: Constructs complete probe sequence from components

**Process**:
1. **Load Encoding Weights**: Reads `E_constrained.csv` from CIPHER training results (gene × bit matrix)
2. **Load Encoding Sites**: Loads database of available probe sequences for target genes (e.g., `mm10_probes_Oct28_2022.csv`)
3. **Load Readout Sequences**: Loads optimized 30-mer readout arm sequences from `optimized_sequences_30mers.csv` (20mers are not used in current workflow)
4. **Filter Readout Sites**: Removes readout sequences containing restriction sites (NotI-HF: `GCGGCCGC`)
5. **Assign Readout Arms to Bits**: 
   - For bits already used in previous designs, reuses existing arm pairs
   - For new bits, randomly selects available arm pairs
   - Ensures no arm is used twice
6. **Convert Weights to Probes**:
   - For each gene and bit, selects probes based on encoding weight
   - Randomly samples probes from available pool for each gene-bit combination
   - Assigns bit number to each selected probe
7. **Assemble Probe Sequences**:
   - Maps readout arm IDs to sequences
   - Creates binding sequences (reverse complement of readout arms)
   - Assembles full probe: `Forward_Primer + arm_0_Binding + encoding_sequence + arm_1_Binding + Restriction_Site + Reverse_Primer_rc`
8. **Output**: Saves complete probe set to `CIPHER_18Bit_WMB_Probes.csv` (or similar name)

**Probe Assembly Structure**:
```
[Forward_Primer] + [arm_0_Binding] + [encoding_sequence] + [arm_1_Binding] + [Restriction_Site] + [Reverse_Primer_rc]
     T7 (20bp)        (30bp)            (30bp)              (30bp)          NotI-HF (8bp)        (20bp)
```

**Components**:
- **Forward_Primer**: T7 promoter sequence (`TAATACGACTCACTATAGGG`)
- **arm_0_Binding**: Reverse complement of first readout arm (binds to bit-specific readout probe)
- **encoding_sequence**: Gene-specific 30-mer sequence from encoding site database
- **arm_1_Binding**: Reverse complement of second readout arm (binds to bit-specific readout probe)
- **Restriction_Site**: NotI-HF site (`GCGGCCGC`) for cloning
- **Reverse_Primer_rc**: Reverse complement of reverse primer (`GCTAGCATGTCTTGACCGCG`)

### Data Files

#### `optimized_sequences_30mers.csv`
**Currently Used**: Optimized 30-mer readout arm sequences used to map bits to genes in probe assembly.

Properties:
- `ID`: Unique identifier (e.g., ZEH0, ZEH1, ...)
- `Sequence`: The 30-mer DNA sequence
- `Arm1`: Alternative arm sequence (often same as Sequence)
- `secondary`: Secondary structure energy (kcal/mol)
- `rc_secondary`: Reverse complement secondary structure energy
- `dimer`: Dimer formation energy with reverse complement
- `rc_dimer`: Reverse complement dimer energy
- `readout_off_target`: Off-target binding score
- `sequence_off_target`: Sequence-level off-target score
- `transcriptome_off_target`: Transcriptome-wide off-target binding score

**Usage**: Each bit in the CIPHER design is assigned a unique pair of 30-mer arms (arm_0 and arm_1) from this file. These arms enable bit-specific detection by binding to complementary readout probes.

#### `optimized_sequences_20mers.csv`
**Future Use**: Optimized 20-mer sequences designed for bDNA (branched DNA) amplification applications.

**Note**: These sequences are **not currently used** in the probe assembly workflow. They are prepared for future experimental protocols that will utilize bDNA amplification technology.

Same structure as 30mers but with 20-mer sequences.

#### `mm10_probes_Oct28_2022.csv`
**Input Database**: Comprehensive database of validated 30-mer probe sequences for mouse (mm10) genome.

**Purpose**: This file contains the pool of available encoding sequences from which probes are selected during assembly. The `make_fasta.ipynb` notebook filters this database to genes used in the CIPHER design and randomly samples probes based on encoding weights.

**Structure**:
- `chrom`, `start`, `end`: Genomic coordinates of the probe sequence
- `seq`: The 30-mer DNA sequence
- `tm`: Melting temperature (°C)
- `onscore`: On-target binding score (0-100)
- `offscore`: Off-target binding score
- `repeat`: Repeat element annotation
- `prob`: Probability score
- `maxkmer`: Maximum k-mer score
- `strand`: Strand orientation (+ or -)
- `gname`: Gene name
- `transcripts`: Number of transcripts
- `len`: Sequence length (30)
- `gc`: GC content (0-1)

**Usage**: The `filter_csv_to_df()` function in `make_fasta.ipynb` filters this database to only include probes for genes present in the CIPHER encoding weights matrix. The `optimize_weight_matrix_to_fasta()` function then selects specific probes from this filtered set based on the encoding weights.

#### `CIPHER_18Bit_WMB_Probes.csv` (Example Output)
Final assembled probe set with columns:
- `chrom`, `start`, `end`: Genomic coordinates of encoding sequence
- `seq`: The 30-mer encoding sequence
- `tm`, `onscore`, `offscore`: Melting temperature and on/off-target scores
- `gname`: Gene name
- `bit`: Bit assignment (0 to n_bit-1)
- `arm_0_ID`, `arm_1_ID`: Readout arm identifiers
- `arm_0_Probe_Sequence`, `arm_1_Probe_Sequence`: Readout arm sequences
- `arm_0_Binding_Sequence`, `arm_1_Binding_Sequence`: Reverse complement binding sequences
- `Forward_Primer_ID`, `Forward_Primer_Sequence`: T7 primer information
- `Reverse_Primer_ID`, `Reverse_Primer_Sequence`, `Reverse_Primer_rc`: Reverse primer information
- `Restriction_site`, `Restriction_site_Sequence`: NotI-HF restriction site
- `Probe_Sequence`: Complete assembled probe sequence (~138 bp)

## Workflow Summary

```
CIPHER Training
    ↓
E_constrained.csv (encoding weights: gene × bit)
    ↓
make_fasta.ipynb
    ├─ Load encoding weights
    ├─ Load encoding sites database
    ├─ Load optimized readout sequences
    ├─ Assign readout arms to bits
    ├─ Convert weights to probe selections
    └─ Assemble complete probes
    ↓
Probes.csv (ready for synthesis)
```

### Step-by-Step Process

1. **Generate Readout Arms** (Run once, reuse for multiple designs):
   - Execute `super_probe_30mer_arms_2025Apr24.ipynb` to generate 30-mer readout arms (currently used)
   - Optionally execute `super_probe_20mer_arms_2025Apr30.ipynb` for future bDNA amplification sequences (not currently used)
   - Generates optimized sequences meeting experimental constraints
   - Output: `optimized_sequences_30mers.csv` (required) and optionally `optimized_sequences_20mers.csv` (for future use)

2. **Assemble Probes** (Run for each CIPHER design):
   - Execute `make_fasta.ipynb`
   - Update paths to:
     - CIPHER encoding weights (`E_constrained.csv`)
     - Encoding sites database (`mm10_probes_Oct28_2022.csv` - now included in this directory)
     - Optimized readout sequences (`optimized_sequences_30mers.csv`)
   - Notebook automatically:
     - Filters encoding sites to genes in design
     - Assigns readout arms to bits
     - Converts weights to probe selections
     - Assembles complete sequences
   - Output: `CIPHER_*Bit_*_Probes.csv`

## Key Design Constraints

### Readout Arm Selection (30-mers, Currently Used)
- **Length**: 30 nucleotides
- **GC Content**: 25-75% of sequence length
- **Melting Temperature**: Target ±5°C window (typically ~60°C)
- **Secondary Structure**: Minimize self-folding (low MFE)
- **Dimer Formation**: Minimize interaction with reverse complement
- **Off-Target Binding**: Minimize binding to transcriptome
- **Restriction Sites**: Avoid NotI-HF site (`GCGGCCGC`) and its reverse complement
- **Function**: Map bits to genes - each bit gets a unique pair of 30-mer arms (arm_0 and arm_1)

### bDNA Amplification Sequences (20-mers, Future Use)
- **Length**: 20 nucleotides
- **Purpose**: Designed for future bDNA amplification applications
- **Status**: Not currently used in probe assembly workflow
- **Same constraints**: GC content, melting temperature, secondary structure, off-target binding

### Probe Assembly
- **Encoding Sequence**: 30-mer from validated probe database
- **Readout Arms**: Two unique 30-mer arms per bit (arm_0 and arm_1) from `optimized_sequences_30mers.csv`
- **Primers**: T7 forward primer and TreeDPNMF reverse primer
- **Restriction Site**: NotI-HF for cloning compatibility
- **Total Length**: ~138 bp per probe

## Dependencies

- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical operations
- **BioPython**: Sequence utilities (MeltingTemp)
- **NUPACK**: Secondary structure and binding energy calculations
- **torch**: Tensor operations for sequence matching
- **tqdm**: Progress bars
- **matplotlib/seaborn**: Visualization (optional)

## Usage Notes

1. **Readout Arm Generation**: This is computationally intensive (hours to days) but only needs to be done once. The generated sequences can be reused across multiple CIPHER designs.

2. **Bit Assignment**: The `make_fasta.ipynb` notebook attempts to reuse readout arms from previous designs when possible, ensuring consistency across experiments.

3. **Probe Selection**: The conversion from encoding weights to specific probes uses random sampling. For reproducibility, set random seeds in the notebook.

4. **File Paths**: Update all file paths in `make_fasta.ipynb` to match your local directory structure and CIPHER training results.

5. **Encoding Site Database**: The `mm10_probes_Oct28_2022.csv` file is included in this directory and contains probe sequences for mouse genes. Ensure it contains probe sequences for all genes in your CIPHER design. For other organisms, you may need to provide a similar database file.

## Output File Format

The final `Probes.csv` file contains one row per probe with:
- Genomic information (chromosome, coordinates)
- Gene and bit assignments
- All sequence components (encoding, arms, primers, restriction sites)
- Complete assembled probe sequence

This file is ready for:
- DNA synthesis ordering
- Experimental validation
- Further analysis or filtering

