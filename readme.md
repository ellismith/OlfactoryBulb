# Olfactory Bulb Model

This repository contains code and analysis for simulating the olfactory bulb network. It is based on the original [JustasB/OlfactoryBulb](https://github.com/JustasB/OlfactoryBulb) repository, with updates and reorganizations to improve usability and analysis workflows.

---

## Folder Structure

```
OlfactoryBulb-1/
├── scripts/                 # All Python scripts
│   ├── stft.py
│   ├── wavelet.py
│   ├── lfp_analysis.py
│   └── ...                  # Other analysis and utility scripts
├── notebooks/               # Jupyter notebooks for exploration and analysis
│   ├── fitting/             # Fitting and validation notebooks
│   │   ├── fitting-GC.ipynb
│   │   ├── fitting-TC.ipynb
│   │   └── ...
│   └── ...                  # Other notebooks
├── results_v1_subcircuit/   # Simulation results and output files
├── readme_figs/             # Figures used in this README
└── ...                      # Other folders (prev_ob_models, olfactorybulb, etc.)
```

---

## Quick Start

1. Clone this repository and enter the folder:

```bash
git clone <your-fork-url>
cd OlfactoryBulb-1
```

2. Install required Python packages (recommended in a virtual environment):

```bash
pip install -r requirements.txt
```

3. Add the `scripts/` folder to your Python path in notebooks:

```python
import sys
sys.path.append("../scripts")
```

4. Run the notebooks in `notebooks/` for analysis, or use scripts in `scripts/` for batch processing.

---

## Example Workflow

![Workflow](workflow.png)

This figure shows the main workflow for simulation, analysis, and figure generation. Additional figures and tables describing results will be added here as the project progresses.

---

## Notes

- Notebooks in `notebooks/fitting/` are organized for parameter fitting and validation.
- `results_v1_subcircuit/` contains simulation outputs; large files such as `.pkl` and `.smr` are included as examples.
- All Python scripts are in `scripts/` and are referenced in notebooks via `sys.path.append("../scripts")`.

