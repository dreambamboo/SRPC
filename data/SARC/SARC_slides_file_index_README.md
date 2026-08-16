# SARC Slides File Index

Generated: 2026-07-21 23:07:01 +08:00
Source table: Z:\data\TCGA-sarcoma_235整理_long.csv
Index file: Z:\data\SARC_slides_file_index.csv

This index was created from the TCGA sarcoma long-format metadata table. The source table contains 235 rows with columns subtype, subtype_index, and ilename.

## Summary

- Total indexed slides: 235
- Expected file type: .svs
- File names in the source table did not include .svs; this index adds .svs to ile_name and elative_path.
- File sizes and source modification times are blank because this index was created from metadata, not by scanning a slide source directory.

## Subtype Counts

- DDLPS: 54
- LMS: 96
- MFS: 22
- MPNST: 9
- SS: 10
- UPS: 44

## Status Columns

- copied_to_project: set to yes after the file has been copied into this project.
- copied_batch: batch name or number, for example atch_001.
- copied_date: date copied, for example 2026-07-21.
- copied_destination: path inside this workspace after copying, usually under Z:\data\SARC.
- processed: set to yes after downstream processing is complete.
- processed_date: date processing finished.
- esult_location: output path, usually under Z:\results.
- 
otes: free-form notes for missing files, source lookup issues, quality issues, or special handling.

## Recommended Workflow

1. Use SARC_slides_file_index.csv to decide which SARC slides to copy in each batch.
2. After copying files into Z:\data\SARC, update copied_to_project, copied_batch, copied_date, and copied_destination.
3. After extracting features, update processed, processed_date, and esult_location.
4. Keep generated outputs in Z:\results, not in Z:\data.