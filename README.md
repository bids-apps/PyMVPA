# PyMVPA BIDS-App
This pipeline is developed at Stanford University and University of California, Irvine, for open-source software distribution.<br /><br />
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1343531.svg)](https://doi.org/10.5281/zenodo.1343531)
## Description
This pipeline takes fMRI data and generates ROI-based & searchlight MultiVariate Pattern Analysis (MVPA) results (including visualized patterns), and also runs Representational Similarity Analysis (RSA) using functionality from PyMVPA. Before running PyMVPA BIDS-App, data needs to be pre-processed using fMRIPrep. The following describes two analysis levels of this app:
- **Participant Prep**: For each subject, takes pre-processed functional images from fMRIPrep, and concatenates them into a single NIfTI file that will be placed under the subject folder in the output directory (/bids_dataset/derivatives/pymvpa/sub-ID). It also creates a text file next to each NIfTI for each subject which contains a list of numbers showing the number of time points in each run.<br /><br />
In addition, an empty "masks" folder under the output folder (/bids_dataset/derivatives/pymvpa/masks) is created at this step which should be filled with NIfTI ROIs before moving to the next step. The "Generating Masks in FSL" section below demonstrates how to generate such masks.<br /><br />
- **Participant Test**: For each subject, iterates through all the ROIs inside the "masks" folder and performs SVM classification, or generates Representational Dissimilarity Matrices (RDM). This step reports results through HTML outputs for each subject, together with visualized patterns in the NIfTI format.
### Generating Masks in FSL
<a href="http://andysbrainblog.blogspot.com/2012/11/creating-masks-in-fsl.html">Here</a> is one approach to creating masks in FSL. If the resolution of your masks doesn't match your functional images, you might need to reslice the masks. Here is one way to do this:<br /><br />
In Python, import the following:
```
from nilearn.image import resample_to_img
import nibabel as nib
```
Next, it's time to load your source (mask) and target (functional) images:
```
src_img = nib.load('mask_name.nii')
trgt_img = nib.load('sub-1_task-tsk_run-01_bold_space-MNI152NLin2009cAsym_preproc.nii') # using the first run here
output = resample_to_img(src_img, trgt_img, interpolation='nearest')
output.to_filename('mask_name.nii')
```
### More on MVPA
Usually at least 8 runs is needed for MVPA analysis. You may also have two really long runs with long rest periods in the middle so that each run can be cut into pieces. The important point is that you need to make sure the chunks remain independent, or analyze data in a way to make them independent!<br /><br />
## Documentation
For more information on PyMVPA, please visit http://www.pymvpa.org/, or look into the "Usage" section.
## How to report errors
For issues or questions, please post to the PyMVPA mailing list (http://www.pymvpa.org/support.html), the BIDS mailing list (https://groups.google.com/forum/#!forum/bids-discussion), or NeuroStars (https://neurostars.org/) with `pymvpa` tag
## Acknowledgements
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1343531.svg)](https://doi.org/10.5281/zenodo.1343531)<br /><br />
Please cite as:<br />
sajjadtorabian. (2018, August 10). sajjadtorabian/PyMVPA: PyMVPA BIDS App version v1.0.0 (Version v1.0.0). Zenodo.
http://doi.org/10.5281/zenodo.1343531
## Usage
The following shows the command line arguments of PyMVPA BIDS-App:
```
usage: run.py [-h] [-k TASK]
              [-c CONDITIONS_TO_CLASSIFY [CONDITIONS_TO_CLASSIFY ...]]
              [-p PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]] [-l NOINFOLABEL]
              [-d [POLY_DETREND]] [-z [ZSCORE]] [-o] [-f FEATURE_SELECTION]
              [-t NFOLD_PARTITIONER] [--skip_bids_validator] [-v]
              bids_dir output_dir {participant_prep,participant_test}

BIDS PyMVPA App

positional arguments:
  bids_dir              The directory with the input dataset formatted
                        according to the BIDS standard.
  output_dir            The directory where the output files should be stored.
  {participant_prep,participant_test}
                        Level of the analysis that will be performed. Multiple
                        participant level analyses can be run independently
                        (in parallel) using the same output_dir.

optional arguments:
  -h, --help            show this help message and exit
  -k TASK, --task TASK  Task to analyze. This has to be specified for both
                        participant_prep and participant_test analysis levels.
  -c CONDITIONS_TO_CLASSIFY [CONDITIONS_TO_CLASSIFY ...], --conditions_to_classify CONDITIONS_TO_CLASSIFY [CONDITIONS_TO_CLASSIFY ...]
                        Conditions to classify.
  -p PARTICIPANT_LABEL [PARTICIPANT_LABEL ...], --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        The label(s) of the participant(s) that should be
                        analyzed. The label corresponds to
                        sub-<participant_label> from the BIDS spec (so it does
                        not include "sub-"). If this parameter is not provided
                        all subjects should be analyzed. Multiple participants
                        can be specified with a space separated list.
  -l NOINFOLABEL, --noinfolabel NOINFOLABEL
                        When building a sample attribute array from an event
                        list, this will be the condition label to assign to
                        all samples for which no stimulation condition
                        information is contained in the events. For more
                        information, look into PyMVPA's events2sample_attr.
                        (default: 'rest')
  -d [POLY_DETREND], --poly_detrend [POLY_DETREND]
                        Order of the Legendre polynomial to remove from the
                        data. This will remove every polynomial up to and
                        including the provided value. If this parameter is not
                        provided no detrending will be performed. (default: 1)
  -z [ZSCORE], --zscore [ZSCORE]
                        Feature-wise, chunk-wise Z-scoring of the data. Scales
                        all features into approximately the same range, and
                        also removes their mean. The argument will specify the
                        condition samples of the dataset used to estimate mean
                        and standard deviation. If this parameter is not
                        provided no normalization will be performed. (default:
                        'rest')
  -o, --condition_attr_onset
                        A sequence of multiple attribute names. All
                        combinations of unique values of the attributes will
                        be used as conditions in HRF modeling. Providing this
                        parameter will add 'onset' to the default ('targets',
                        'chunks') so that one estimate per each individual
                        event is produced (more, but noisier estimates) rather
                        than a separate model for each condition for each run.
                        This is a trade off between number of training samples
                        and noise reduction.
  -f FEATURE_SELECTION, --feature_selection FEATURE_SELECTION
                        Uses an ANOVA measure to select features with the
                        highest F-scores. Will perform
                        FixedNElementTailSelector if f > 1, and
                        FractionTailSelector if f < 1. If this parameter is
                        not provided no feature selection will be performed.
  -t NFOLD_PARTITIONER, --nfold_partitioner NFOLD_PARTITIONER
                        When performing cross-validation on a dataset with n
                        chunks, with t = 1 (default), it would generate n
                        partition sets, where each chunk is sequentially taken
                        out to form a second partition, while all other
                        samples together form the first partition. If t > 1,
                        then all possible combinations of t number of chunks
                        are taken out. If t is a float between 0 and 1, it
                        specifies the ratio of present unique values to be
                        taken.
  --skip_bids_validator
                        Whether or not to perform BIDS dataset validation
  -v, --version         show program's version number and exit
```
The following shows how to run the app in prep mode for an "objectviewing" task and participants 1 & 2:
```
docker run -i --rm \
	-v [path to BIDS root]:/bids_dataset:ro \
	-v [path to BIDS root/derivatives/pymvpa]:/outputs \
	bids/pymvpa \
	/bids_dataset /outputs participant_prep -k objectviewing -p 1 2
```
And, here is how to run the app in test mode, for the classification of "face"s versus "house"s, with detrending of time-series and z-scoring applied. Note that the same output folder as above should be used:
```
docker run -i --rm \
	-v [path to BIDS root]:/bids_dataset:ro \
	-v [path to BIDS root/derivatives/pymvpa]:/outputs \
	bids/pymvpa \
	/bids_dataset /outputs participant_test -k objectviewing -c face house -p 1 2 -d -z
```
## Special Considerations
Note that functional runs get concatenated into single NIfTI files for each subject, so be mindful of your memory capacity.
