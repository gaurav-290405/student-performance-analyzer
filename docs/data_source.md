# Data source note
**Primary dataset:** UCI Student Performance (mathematics), `student-mat.csv`, 395 rows x 33 columns, separator `;`.
Download: https://archive.ics.uci.edu/dataset/320/student+performance  -> put the file at `data/raw/student-mat.csv`.
Verify the licence on the source page before use.

**Fallback (used when the real file is absent):** `src/data_generation.py` makes synthetic data with the same columns.
Assumptions: grades follow a latent "ability" plus study time (+), past failures (-), absences (-), going out (-),
higher-education aim (+); about 3% of students score 0 (dropouts). Effects are illustrative, not real findings.
