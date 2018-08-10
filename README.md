# PyMVPA BIDS App
## Description
This pipeline takes fMRI data and generates ROI based MultiVariate Pattern Analysis (MVPA) outputs (detailed classification results + visualized patterns) using functionality from FSL and PyMVPA. Before running PyMVPA BIDS, you need to preprocess data using fmriprep. When you're ready, run the app at its two analysis levels in order:
- **participant_prep**: For each subject, takes preprocessed functional runs from fmriprep under derivatives, and concatenates them into single NIfTI files (e.g. sub-1_task-objectviewing_bold_space_preproc.nii.gz) that will be placed in separate subject folders under the output folder (e.g. /bids_dataset/derivatives/pymvpa) specified by the user. It also creates a text file (e.g. sub-1_task-objectviewing_dim4.txt) next to each NIfTI for each subject which is a list containing all the runs' number of time points. These two files will then be used in participant_test. In addition, an empty "masks" folder under the output folder (in our example: /bids_dataset/derivatives/pymvpa/masks) is created in this step which before getting to participant_test should be filled with NIfTI ROIs that you want MVPA to run on. If you don't have your masks already and want to know how to generate one, please read the "Generating Masks in FSL" section below.
- **participant_test**: For each subject, iterates through all the ROIs inside the "masks" folder and performs SVM classification over HRF modeling of specified conditions. It generates HTML outputs containing detailed classification results, plus visualized patterns (e.g. /bids_dataset/derivatives/pymvpa/sub-1/sub-1_task-objectviewing_Occipital_Fusiform_Gyrus_face_house_pattern.nii.gz).
### Generating Masks in FSL
When you create a mask in FSL using 1mm or 2mm MNI atlases,
http://andysbrainblog.blogspot.com/2012/11/creating-masks-in-fsl.html (src_img)
you need to reslice it to 3.5 x 3.75 x 3.75 resolution to match fmriprep output. Here is our way of doing this:
Open a terminal and start IPython:
```
ipython
```
In IPython:
```
from nilearn.image import resample_to_img
import nibabel as nib
```
Next, it is time to load your source (image to resample) and target (reference image taken for resampling) images:
```
src_img = nib.load('Temporal_Occipital_Fusiform_Cortex.nii')
trgt_img = nib.load('sub-1_task-objectviewing_run-01_bold_space-MNI152NLin2009cAsym_preproc.nii') # this could be any of the preprocessed NIfTIs
output = resample_to_img(src_img, trgt_img, interpolation='nearest')
output.to_filename('Temporal_Occipital_Fusiform_Cortex.nii') # used the same name to replace
```
## Documentation
For more information on PyMVPA, please visit http://www.pymvpa.org/, or look into the "Usage" section and comments inside run.py!
## How to report errors
For issues or questions, please post to the PyMVPA mailing list (http://www.pymvpa.org/support.html), the BIDS mailing list (https://groups.google.com/forum/#!forum/bids-discussion), or NeuroStars (https://neurostars.org/) with pymvpa tag
## Acknowledgements
## Usage
This App has the following command line arguments:
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
To run it in participant_prep level mode (for participants 1 and 2):
```
docker run -i --rm \
	-v /Users/Sajjad/Visual_object_recognition:/bids_dataset:ro \
	-v /Users/Sajjad/Visual_object_recognition/derivatives/pymvpa:/outputs \
	sajjadtorabian/pymvpa \
	/bids_dataset /outputs participant_prep -k objectviewing -p 1 2
```
To run it in participant_test level mode (should use the same output folder as above):
```
docker run -i --rm \
	-v /Users/Sajjad/Visual_object_recognition:/bids_dataset:ro \
	-v /Users/Sajjad/Visual_object_recognition/derivatives/pymvpa:/outputs \
	sajjadtorabian/pymvpa \
	/bids_dataset /outputs participant_test -k objectviewing -c face house -p 1 2 -d -z
```
## Special considerations
