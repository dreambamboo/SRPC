# Data Metadata

This folder stores metadata only; raw WSI files and extracted feature matrices are not included.

- `SARC/`: TCGA-SARC slide index, subtype labels, survival manifest example, and summary.
- `MESO/`: TCGA-MESO slide metadata and survival manifest example.
- `splits/SARC/5fold_pfi_seed2026`: fixed five-fold PFI split used for Table 1 verification.
- `splits/MESO/5fold_pfi_seed2026`: prepared MESO five-fold split metadata for external experiments.

Download raw TCGA slides and clinical metadata from GDC:

- TCGA-SARC project page: https://portal.gdc.cancer.gov/projects/TCGA-SARC
- TCGA-MESO project page: https://portal.gdc.cancer.gov/projects/TCGA-MESO
- GDC Data Portal repository: https://portal.gdc.cancer.gov/repository
- GDC Data Transfer Tool: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

The manifest example records the expected feature path columns for `resnet50_trunc`, `uni_v2`, and `conch_v1_5`. When reproducing experiments on another machine, copy the example manifest outside the repository and update these columns to point to your extracted `.pt` feature files:

```text
resnet50_trunc_pt_path
uni_v2_pt_path
conch_v1_5_pt_path
```

The inference scripts read these paths directly; raw slides and extracted features should not be copied into this open-source code folder.
