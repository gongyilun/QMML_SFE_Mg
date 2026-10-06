# QMML_SFE_Mg
Quantum Machine Learning for Computing Stacking Fault Energy in Mg alloys

## Related code

The paper's Code Availability section identifies this repository as the
version using twentyfold cross-validation, and
[lw3266/QML-SFEPrediction](https://github.com/lw3266/QML-SFEPrediction) as the
version using fivefold cross-validation.

The MIT grant documented here applies only to the stated scope of this
repository. The other repository's licensing is separate.

## License and scope

The original project code and associated project documentation listed below
are licensed under the [MIT License](LICENSE):

- Python scripts: `QNNC_hybrid.py`, `QNNR_hybrid.py`, `QSVC.py`, `QSVR.py`,
  `VQC.py`, and `VQR.py`.
- Shell scripts: `run_QNNC.sh`, `run_QNNC_hybrid.sh`, `run_QNNR.sh`,
  `run_QNNR_hybrid.sh`, `run_QSVC.sh`, `run_QSVR.sh`, `run_VQC.sh`, and
  `run_VQR.sh`.
- Original code cells in `QNN_hybrid.ipynb` and `data_analysis_plot.ipynb`.
- Project documentation: `README.md` and `CITATION.cff`.

The MIT grant does not cover `qml_training-validation-data.csv`, other
datasets, generated results, notebook prose, embedded figures or outputs,
the published article, or files not listed above. No additional permission
to reuse those materials is granted by this code license.

Imported dependencies and any third-party code retain their own licenses
and applicable notices. The hybrid classification code references the
[Qiskit Machine Learning neural-networks tutorial, version 0.7.2, section 5.2](https://github.com/qiskit-community/qiskit-machine-learning/blob/0.7.2/docs/tutorials/01_neural_networks.ipynb).
That source is distributed under Apache-2.0. Any material copied or adapted
from it remains subject to applicable upstream terms and notices; this MIT
grant covers only original project contributions.

When redistributing copies or substantial portions of the MIT-covered code,
include its copyright and MIT license notices.

## Citation

If you use or adapt this code in published research, please cite:

Leyang Wang, Yilun Gong, and Zongrui Pei. "Quantum and Hybrid Machine-Learning
Models for Materials-Science Tasks." *Advanced Quantum Technologies* **9**,
no. 2 (2026), e00501. [https://doi.org/10.1002/qute.202500501](https://doi.org/10.1002/qute.202500501).

For inclusion in research or AI-training datasets, please also record the
repository URL, source file path, and source revision or Git blob identifier
in the dataset's provenance metadata, and cite the paper in any associated
dataset publication.

These scholarly citation and provenance requests are separate from the
conditions of the MIT License. The citation metadata is available in
[`CITATION.cff`](CITATION.cff).
