# Reference outputs

This directory contains the numerical tables and figures produced by the
offline pipeline. They are committed as a reviewable reference run.

The verified reproducibility target is the reported numerical content:
weekly metrics, bootstrap intervals, feature-importance and SHAP tables,
validation diagnostics, map thresholds, classified GeoTIFFs and operational
counts. These reproduced under Python 3.14.3, scikit-learn 1.9.0 and numpy
2.4.2.

PNG bytes are not a cross-environment reproducibility target. SHAP beeswarm
jitter, font discovery, antialiasing and image metadata can vary with SHAP,
Matplotlib, Pillow and platform versions even when the underlying SHAP values
and every reported model metric are unchanged. Compare the CSVs before treating
a cosmetic image diff as a result change.

`requirements-lock.txt` pins the model-sensitive numerical packages but is not
a complete frozen plotting environment. Capture a full `pip freeze` if exact
rendering of a newly generated figure must be archived.
