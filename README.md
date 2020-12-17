# brain_age

Predict age from brain grey matter (regression).

## Preprocessed input data

Voxel Based Morphometry (VBM) from [cat12](http://www.neuro.uni-jena.de/cat/):

- ROIs of Gray Matter (GM) scaled for the Total Intracranial Volume (TIV)
  * `[test|train]_rois.csv` 284 features

- VBM map in the MNI space (3 D map)
  * `[test|train]_train_vbm.npz` 3D images of shapes (121, 145, 121).
  This npz contains the 3D mask and the affine transformation to MNI
  referential. Masking the brain provide 331 695 input features (voxels).

Input data is the concatenation of 284 ROIs features with 331 695 features.
Those two blocks are redundant. To select only on ROIs features do:

```
x_arr[:, :284]
```

To select only on VBM (voxel with the brain) features do:

```
x_arr[:, 284:]
```

There are 357 samples in the training set and 90 samples in the test set.

## Target

- `[test|train]_participants.csv`:
   * `age` for regression problem.

## Links


- [RAMP-workflow’s documentation](https://paris-saclay-cds.github.io/ramp-workflow/)
- [RAMP-workflow’s github](https://github.com/paris-saclay-cds/ramp-workflow)
- [RAMP Kits](https://github.com/ramp-kits)

## Getting started

This starting kit requires Python and the following dependencies:

* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `matplolib`
* `seaborn`
* `nilearn`
* `jupyter`
* `ramp-workflow`

Therefore, we advise you to install [Anaconda
distribution](https://www.anaconda.com/download/) which include almost all
dependencies.

Only `nilearn` and `ramp-workflow` are not included by default in the Anaconda
distribution. They will be installed from the execution of the notebook.

Execute the jupyter notebook, from the root directory using:

```
jupyter notebook brain_age_starting_kit.ipynb
```


## Advanced install using `conda` (optional)

We provide both an `environment.yml` file which can be used with `conda` to
create a clean environment and install the necessary dependencies.

```
conda env create -f environment.yml
```

Then, you can activate the environment using:

```
source activate brain_age
```

for Linux and MacOS. In Windows, use the following command instead:

```
activate brain_age
```
