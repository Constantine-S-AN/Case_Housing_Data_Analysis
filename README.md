## Case Housing — Data Cleaning

This repo contains a simple data-cleaning script for the Ames/Case Housing dataset.

### Files
- **`train.csv`**: original training data
- **`train_cleaned.csv`**: cleaned output produced by the script
- **`clean_data.py`**: cleaning script
- **`data_description.txt`**: feature descriptions

### How to run

```bash
python clean_data.py
```

This will write `train_cleaned.csv` in the same folder.

### Graphs and Analysis
See `graph/graph analysis.md` for detailed interpretations of each plot.

### Reproduce Report

Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_model_report.py
```

Final submission file:
- `outputs/Model_Report.pdf`
- or `outputs/Model_Report.html` (open in browser and print/save as PDF)
