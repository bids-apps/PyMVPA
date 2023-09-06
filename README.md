# PyMVPA BIDS-App

This pipeline is developed at Stanford University and University of California, Irvine, for open-source software distribution.<br /><br />
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1343531.svg)](https://doi.org/10.5281/zenodo.1343531)

## Citation

Please cite Torabian et al. (2023) [<a href="https://www.frontiersin.org/articles/10.3389/fnins.2023.1233416/full">paper</a>]

## Description

This pipeline takes fMRI data and generates ROI-based & searchlight MultiVariate Pattern Analysis (MVPA) results (including visualized patterns), and also runs Representational Similarity Analysis (RSA) using functionality from PyMVPA. Before running PyMVPA BIDS-App, data needs to be pre-processed using fMRIPrep. 

The following describes two analysis levels of this app:
- **Participant Prep**: For each subject, takes pre-processed functional images from fMRIPrep, and concatenates them into a single NIfTI file that will be placed under the subject folder in the output directory (/bids_dataset/derivatives/pymvpa/sub-ID). It also creates a text file next to each NIfTI for each subject which contains a list of numbers showing the number of time points in each run.<br /><br />
In addition, an empty "masks" folder under the output folder (/bids_dataset/derivatives/pymvpa/masks) is created at this step which should be filled with NIfTI ROIs before moving to the next step. The "Generating Masks in FSL" section below demonstrates how to generate such masks.<br /><br />
- **Participant Test**: For each subject, iterates through all the ROIs inside the "masks" folder and performs SVM classification, or generates Representational Dissimilarity Matrices (RDM). This step reports results through HTML outputs for each subject, together with visualized patterns in the NIfTI format.

### Generating Masks in FSL

<a href="http://andysbrainblog.blogspot.com/2012/11/creating-masks-in-fsl.html">Here</a> is one approach to creating masks in FSL. If the resolution of your masks doesn't match your functional images, you might need to reslice the masks. Here is one way to do this:<br /><br />

In Python, import the following:
```python
from nilearn.image import resample_to_img
import nibabel as nib
```

Next, it's time to load your source (mask) and target (functional) images:
```python
src_img = nib.load('mask_name.nii')
trgt_img = nib.load('sub-1_task-tsk_run-01_bold_space-MNI152NLin2009cAsym_preproc.nii') # using the first run here
output = resample_to_img(src_img, trgt_img, interpolation='nearest')
output.to_filename('mask_name.nii')
```

### More on MVPA

Usually at least 8 runs is needed for MVPA analysis. You may also have two really long runs with long rest periods in the middle so that each run can be cut into pieces. The important point is that you need to make sure the chunks remain independent, or analyze data in a way to make them independent!<br /><br />
## Documentation
For more information on PyMVPA, please visit http://www.pymvpa.org/, or look into the "Usage" section below.

## How to report errors

For issues or questions, please post to the PyMVPA mailing list (http://www.pymvpa.org/support.html), the BIDS mailing list (https://groups.google.com/forum/#!forum/bids-discussion), or NeuroStars (https://neurostars.org/) with `pymvpa` tag

## Usage

The following shows the command line arguments of PyMVPA BIDS-App:

```
usage: run.py [-h] [-p PARTICIPANT_ID [PARTICIPANT_ID ...]] [-s SESSION]
              [--searchlight [SEARCHLIGHT]] [-t TASK]
              [-c CONDITIONS_TO_CLASSIFY [CONDITIONS_TO_CLASSIFY ...]]
              [--noinfolabel [NOINFOLABEL]] [--poly_detrend [POLY_DETREND]]
              [--tzscore] [--bzscore] [-i] [-f FEATURE_SELECT]
              [--cvtype CVTYPE] [--lss] [--rsa] [--surf]
              [--space [{fsnative,fsaverage}]] [--hemi [{l,r}]] [--mask MASK]
              [--dist [{correlation,euclidean,mahalanobis}]] [--nproc [NPROC]]
              [--skip_bids_validator] [-v]
              bids_dir output_dir {participant_prep,participant_test}

PyMVPA BIDS-App

positional arguments:
  bids_dir              The input directory with dataset formatted according
                        to BIDS standard.
  output_dir            The directory that will store outputs.
  {participant_prep,participant_test}
                        Level of the analysis that will be performed. Multiple
                        analyses can run independently in parallel.

optional arguments:
  -h, --help            show this help message and exit
  -p PARTICIPANT_ID [PARTICIPANT_ID ...], --participant_id PARTICIPANT_ID [PARTICIPANT_ID ...]
                        Subjects (specified by ID) that will be analyzed. The
                        ID corresponds to sub-<participant_id> from BIDS
                        specification. If specific IDs are not provided, all
                        subjects will be analyzed. Multiple subjects can be
                        specified by a space separated list.
  -s SESSION, --session SESSION
                        Session ID for multi-session datasets.
  --searchlight [SEARCHLIGHT]
                        Performs searchlight analysis with s being the radius
                        of spheres/discs in volumetric/surface mode. If this
                        flag is not enabled, ROI-based analysis will run.
                        (default: 5.0)
  -t TASK, --task TASK  Task to analyze. This has to be specified for both
                        participant_prep and participant_test analysis levels.
  -c CONDITIONS_TO_CLASSIFY [CONDITIONS_TO_CLASSIFY ...], --conditions_to_classify CONDITIONS_TO_CLASSIFY [CONDITIONS_TO_CLASSIFY ...]
                        Conditions to classify.
  --noinfolabel [NOINFOLABEL]
                        When building a sample attribute array from an event
                        list, this will be assigned to all samples with no
                        label. For more information, check PyMVPA's
                        events2sample_attr. (default: 'rest')
  --poly_detrend [POLY_DETREND]
                        Order of Legendre polynomial to remove from the data.
                        This will remove every polynomial up to and including
                        the provided value. If this parameter is not provided
                        no detrending will be performed. (default: 1)
  --tzscore             Feature-wise, run-wise z-scoring of time-series.
                        Scales all features into approximately the same range,
                        and removes their mean. If this parameter is not
                        provided no normalization will be performed.
  --bzscore             Feature-wise z-scoring of GLM beta estimates across
                        all runs.
  -i, --indiv_trials    When (HRF) modeling the time-series, enabling this
                        flag will estimate betas per individual trials, rather
                        than per condition per run. This provides more but
                        noisier estimates. A trade off between the number of
                        training samples and noise reduction should be made by
                        the user.
  -f FEATURE_SELECT, --feature_select FEATURE_SELECT
                        Uses ANOVA to select features with highest F-scores.
                        Will perform FixedNElementTailSelector if f > 1, and
                        FractionTailSelector if f < 1. If this parameter is
                        not provided no feature selection will be performed.
  --cvtype CVTYPE       When running cross-validation on a dataset with n
                        runs, with cvtype = 1 (default), n partition sets will
                        be generated, where each run is sequentially left out
                        to form one partition, with all other runs forming the
                        other partition together. If cvtype > 1, all possible
                        combinations of cvtype number of runs are left out.
  --lss                 Enabling this parameter will run GLM in a Least
                        Squares Single (LSS) fashion. Default is Ordinary
                        Least Squares (OLS).
  --rsa                 Enabling this parameter will run Representational
                        Similarity Analysis. Default is classification.
  --surf                Enabling this parameter will run analyses on surface.
                        Default is volumetric.
  --space [{fsnative,fsaverage}]
                        Surface space of analysis. Options are fsnative or
                        fsaverage.
  --hemi [{l,r}]        Hemisphere of analysis. Need to specify for surface.
                        Options are l (left) or r (right).
  --mask MASK           NIfTI mask used for both ROI-based analysis and
                        searchlight.
  --dist [{correlation,euclidean,mahalanobis}]
                        Distance metric to use for Representational
                        Dissimilarity Matrices (RDM). Choices are correlation
                        (default), euclidean, and mahalanobis.
  --nproc [NPROC]       Number of CPUs to use (default: 8).
  --skip_bids_validator
                        Skipping BIDS validation
  -v, --version         show program's version number and exit
```

The following shows how to run the app in prep mode for task "objectviewing" of participants 1 & 2:
```bash
docker run -i --rm \
	-v [path to BIDS root]:/bids_dataset:ro \
	-v [path to BIDS root/derivatives/pymvpa]:/outputs \
	bids/pymvpa \
	/bids_dataset /outputs participant_prep -p 1 2 -t objectviewing
```

And, here is how to run the app in test mode, for the classification of "face"s versus "house"s, with z-scoring of time-series applied. Note that the same output folder as above should be used:
```bash
docker run -i --rm \
	-v [path to BIDS root]:/bids_dataset:ro \
	-v [path to BIDS root/derivatives/pymvpa]:/outputs \
	bids/pymvpa \
	/bids_dataset /outputs participant_test -p 1 2 -t objectviewing -c face house --bzscore
```

## Special Considerations

Note that functional runs get concatenated into single NIfTI files for each subject, so be mindful of your memory capacity.
