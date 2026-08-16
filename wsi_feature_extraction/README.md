# WSI Feature Extraction

This folder contains the preprocessing and feature-extraction pipeline used by the project. It includes a compact copy of the CLAM components required for tissue segmentation, patch-coordinate generation, and patch feature extraction.

Main entry point:

```bash
python run_clam_feature_pipeline.py --dataset SARC --source_dir /path/to/slides --results_root /path/to/outputs --checkpoints_root /path/to/encoder_checkpoints --encoders resnet50_trunc uni_v2 conch_v1_5 --device cuda
```

Expected encoder checkpoint layout is documented in the root `ReadMe.md`. The pipeline performs an OpenSlide preflight check, skips unreadable slides, writes a recopy list, generates CLAM patch-coordinate h5 files, and extracts GPU features for each selected encoder.

Default WSI setting: patch level 1, patch size 512, step size 512, corresponding to the x20 workflow used in the manuscript. Confirm OpenSlide level mapping before applying the script to a new dataset.
