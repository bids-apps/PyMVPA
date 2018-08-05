# PyMVPA
For more information on PyMVPA, please visit http://www.pymvpa.org/
## Description
## Documentation
## How to report errors
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
## Special considerations
