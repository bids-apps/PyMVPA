#!/usr/bin/env python2.7
import argparse
import os
from glob import glob
from os import listdir
from os.path import join
import mvpa2
import json
import csv
from mvpa2.suite import *
import tempfile
import subprocess

__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'version')).read()

def run_script(script):
    with tempfile.NamedTemporaryFile() as scriptfile:
        scriptfile.write(script)
        scriptfile.flush()
        subprocess.call(['/bin/bash', scriptfile.name])

parser = argparse.ArgumentParser(description='BIDS PyMVPA App')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                    'Multiple participant level analyses can be run independently '
                    '(in parallel) using the same output_dir.',
                    choices=['participant_prep', 'participant_test'])
parser.add_argument('-k', '--task', help='Task to analyze. This has to be specified for both '
                    'participant_prep and participant_test analysis levels.')
parser.add_argument('-c', '--conditions_to_classify', help='Conditions to classify.',
                    nargs="+")
parser.add_argument('-p', '--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                    'corresponds to sub-<participant_label> from the BIDS spec '
                    '(so it does not include "sub-"). If this parameter is not '
                    'provided all subjects should be analyzed. Multiple '
                    'participants can be specified with a space separated list.',
                    nargs="+")
parser.add_argument('-l', '--noinfolabel', help='When building a sample attribute array from an event list, this will '
                    'be the condition label to assign to all samples for which '
                    'no stimulation condition information is contained in the events. '
                    'For more information, look into PyMVPA\'s events2sample_attr. (default: \'rest\')',
                    default='rest')
parser.add_argument('-d', '--poly_detrend', help='Order of the Legendre polynomial to remove from the data. '
                    'This will remove every polynomial up to and including the '
                    'provided value. If this parameter is not provided no detrending '
                    'will be performed. (default: 1)',
                    nargs='?', const=1, type=int)
parser.add_argument('-z', '--zscore', help='Feature-wise, chunk-wise Z-scoring of the data. Scales '
                    'all features into approximately the same range, and also removes '
                    'their mean. The argument will specify the condition samples of '
                    'the dataset used to estimate mean and standard deviation. If this '
                    'parameter is not provided no normalization will be performed. (default: \'rest\')',
                    nargs='?', const='rest')
parser.add_argument('-o', '--condition_attr_onset', help='A sequence of multiple attribute names. All combinations '
                    'of unique values of the attributes will be used as conditions in '
                    'HRF modeling. Providing this parameter will add \'onset\' to the '
                    'default (\'targets\', \'chunks\') so that one estimate per each '
                    'individual event is produced (more, but noisier estimates) rather '
                    'than a separate model for each condition for each run. This is a '
                    'trade off between number of training samples and noise reduction.',
                    action='store_true')
parser.add_argument('-f', '--feature_selection', help='Uses an ANOVA measure to select features with the highest '
                    'F-scores. Will perform FixedNElementTailSelector if f > 1, and '
                    'FractionTailSelector if f < 1. If this parameter is not provided '
                    'no feature selection will be performed.',
                    type=float)
parser.add_argument('-t', '--nfold_partitioner', help='When performing cross-validation on a dataset with n chunks, '
                    'with t = 1 (default), it would generate n partition sets, where '
                    'each chunk is sequentially taken out to form a second partition, '
                    'while all other samples together form the first partition. If t > 1, '
                    'then all possible combinations of t number of chunks are taken out. '
                    'If t is a float between 0 and 1, it specifies the ratio of present '
                    'unique values to be taken.',
                    default=1, type=float)
parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation',
                   action='store_true')
parser.add_argument('-v', '--version', action='version',
                    version='PyMVPA BIDS-App Version {}'.format(__version__))


args = parser.parse_args()

if not args.skip_bids_validator:
    run_script('bids-validator %s' % args.bids_dir)

os.chdir(args.bids_dir)

subjects_to_analyze = []
# only for a subset of subjects
if args.participant_label:
    subjects_to_analyze = args.participant_label
# for all subjects
else:
    subject_dirs = sorted(glob(os.path.join(args.bids_dir, "sub-*")))
    subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]

task_json = json.load(open("task-" + args.task + "_bold.json"))
TR = task_json["RepetitionTime"]

# running participant_prep level
if args.analysis_level == "participant_prep":
    my_bash_script = '''\
    path=%s
    out_path=%s
    task=%s
    mkdir $out_path
    mkdir $out_path/masks
    sub_IDs="%s"
    for subjects in ${sub_IDs}; do
        subjects=$path/derivatives/fmriprep/sub-$subjects
        [ -d "${subjects}" ] || continue
        subj_name=$(basename $subjects)
        mkdir $out_path/$subj_name
        fslmerge -t "$out_path/$subj_name/""$subj_name""_task-""$task""_bold_space_preproc" $subjects/func/*$task*_preproc.nii.gz
        #reading dim4 values of all runs
        for runs in $path/derivatives/fmriprep/$subj_name/func/*$task*_preproc.nii.gz; do
            fslval $runs dim4 >> "$out_path/$subj_name/""$subj_name""_task-""$task""_dim4.txt"
        done
    done
    '''
    my_bash_script = my_bash_script % (args.bids_dir, args.output_dir, args.task, ' '.join(subjects_to_analyze))
    run_script(my_bash_script)

# running participant_test level
elif args.analysis_level == "participant_test":
    for subjects in subjects_to_analyze:
        subj_name = 'sub-' + str(subjects)
        all_runs_bold_fname = os.path.join(args.output_dir, subj_name,
                                           subj_name + '_task-' + args.task + '_bold_space_preproc.nii.gz')

        chunks = []
        duration = []
        onset = []
        targets = []
        chunk_counter = 0
        # looping through all the runs' tsv events files
        for filename in sorted(os.listdir(os.path.join(args.bids_dir, subj_name, 'func'))):
            if filename.endswith(".tsv"):
                with open(os.path.join(args.bids_dir, subj_name, 'func', filename)) as tsvfile:
                    reader = csv.DictReader(tsvfile, dialect='excel-tab')
                    for row in reader:
                        chunks.append(chunk_counter)
                        duration.append(float(row['duration']))
                        onset.append(float(row['onset']))
                        targets.append(row['trial_type'])
                chunk_counter = chunk_counter + 1
                continue
            else:
                continue

        scan_ips_list = []
        chunks_labels = []  # will be used to label samples
        chunk_counter = 0  # labeling starts from zero
        with open(
            os.path.join(args.output_dir, subj_name, subj_name + '_task-' + args.task + '_dim4.txt')) as sips:
            for line in sips:
                scan_ips_list.append(int(line))
                for i in range(0, int(line)):
                    chunks_labels.append(chunk_counter)
                chunk_counter += 1

        number_of_runs = chunk_counter

        offsets = []
        for i in range(0, number_of_runs - 1):  # exp: for 5 runs, we need 4 offsets - we can also use
                                                # len(scan_ips_list) rather than number_of_runs
            if i == 0:
                offsets.append(scan_ips_list[i] * TR)  # scan_ips_list[i] is the number of volumes in chunk i
            else:
                offsets.append(offsets[-1] + scan_ips_list[i] * TR)  # some_list[-1] is the shortest and most Pythonic
                                                                     # way of getting the last element of a list

        # generating a list of dictionaries
        original_events = []
        for i in range(0, len(chunks)):  # we could instead use len(duration), len(onset), or len(targets)
            current_event = {}
            current_event['chunks'] = chunks[i]
            current_event['duration'] = duration[i]
            current_event['onset'] = onset[i]
            if chunks[i] != 0:  # no offset for the first chunk
                current_event['onset'] = current_event['onset'] + offsets[chunks[i] - 1]
            current_event['targets'] = targets[i]
            original_events.append(current_event)

        # events and cond_attr will later be passed to fit_event_hrf_model:
        events = [ev for ev in original_events if ev['targets'] in args.conditions_to_classify]
        if args.condition_attr_onset:
            cond_attr = ('onset', 'targets', 'chunks')
            est = 'Individual Event'  # will be used in the html output
        else:
            cond_attr = ('targets', 'chunks')
            est = 'Condition per each Run'

        clf = SVM()  # SVMs come with sensitivity analyzers!
        # feature selection is enabled
        if args.feature_selection:
            if args.feature_selection > 1:
                fs = 'On (selected %d features)' % args.feature_selection
                fsel = SensitivityBasedFeatureSelection(
                    OneWayAnova(),
                    FixedNElementTailSelector(int(args.feature_selection), mode='select', tail='upper')
                )
            elif args.feature_selection <= 1:
                fs = 'On (selected %.2f%% of features)' % (args.feature_selection*100)
                fsel = SensitivityBasedFeatureSelection(
                    OneWayAnova(),
                    FractionTailSelector(args.feature_selection, mode='select', tail='upper')
                )
            # the following approach uses the full dataset to determine which features show category differences in
            # the whole dataset, including our supposed-to-be independent testing data (precisely constitutes the
            # double-dipping procedure):
            # fsel.train(evds)
            # evds_p = fsel(evds)
            # to implement an ANOVA-based feature selection properly we have to do it on the training dataset only:
            clf = FeatureSelectionClassifier(clf, fsel)
        # no feature selection
        else:
            fs = 'Off'
            pass
        if args.nfold_partitioner >= 1:
            cv_type = int(args.nfold_partitioner)
        else:
            cv_type = args.nfold_partitioner

        # a convenient way to access the total performance of the underlying classifier, and get the sensitivities at
        # the same time: (can effectively perform a cross-validation analysis internally)
        sclf = SplitClassifier(clf, NFoldPartitioner(cvtype=cv_type),
                               # exp: if cvtype=5 and the total number of runs is 12, then we'll
                               # have C(12,5)=12!/((12-5)!*5!) ways of splitting the data, and
                               # will have 5 runs for testing in each combination
                               enable_ca=['stats'])
        cv_sensana = sclf.get_sensitivity_analyzer()  # no post-processing here -> obtaining sensitivity maps from all
                                                      # internally trained classifiers =
                                                      # C(number of conditions_to_classify,2)*number_of_runs maps

        subj_html = open(os.path.join(args.output_dir, subj_name + '.html'), 'w')
        html_str = """
        <!DOCTYPE html>
        <html>
        <body>
                <h1 style="font-family:amatic;">Summary</h1>
                <ul>
                <li>Subject ID: %s</li>
                <li>Functional Series: %s</li>
                <li>Task: %s</li>
                <li>Conditions Classified: %s</li>
                <li>No Info Label: '%s'</li>
                <li>Detrending: %s</li>
                <li>Normalization: %s</li>
                <li>One Estimate per each %s</li>
                <li>Feature Selection: %s</li>
                <li>NFold Partitioner: %s</li>
                </ul>
        </body>
        </html>
        """
        detr = 'On (polyord=%d)' % args.poly_detrend if args.poly_detrend else 'Off'
        norm = 'On (condition sample=\'%s\')' % args.zscore if args.zscore else 'Off'
        html_str = html_str % (subj_name, number_of_runs, args.task, args.conditions_to_classify, args.noinfolabel,
                               detr, norm, est, fs, cv_type)
        subj_html.write(html_str)


        ROIs = []
        ROIs = sorted(os.listdir(os.path.join(args.output_dir, 'masks')))

        for ROIloop in range(0, len(ROIs)):
            ROI_name = ROIs[ROIloop].split('.')[0]

            mask_fname = os.path.join(args.output_dir, 'masks', ROIs[ROIloop])
            fds = fmri_dataset(samples=all_runs_bold_fname,
                               mask=mask_fname)

            # chunks_labels = events2sample_attr(original_events, fds.sa.time_coords, condition_attr='chunks')
            # rather than using events2sample_attr (or assign_conditionlabels) to attribute chunks labels to
            # samples which would be tricky because of noinfolabel, we do:
            fds.sa['chnks'] = chunks_labels  # we call this sample attribute 'chnks' so later it won't be mistaken for
                                             # 'chunks' in events
            targets_labels = events2sample_attr(original_events, fds.sa.time_coords, noinfolabel=args.noinfolabel,
                                                condition_attr='targets')
            fds.sa['trgts'] = targets_labels

            # find_events won't work for this app because we consider the possibility of jittering
            # events = find_events(targets=fds.sa.trgts, chunks=fds.sa.chnks)

            # note that we will be using 'chnks' and 'trgts' sample attributes for detrending and normalization
            # purposes, and not for cross-validation because with jittering, a lot of information
            # will be lost that way -> events will be used rather than samples

            # since our data usually stems from several different runs, the assumption of a continuous linear trend
            # across all runs is not appropriate:
            # poly_detrend(fds, polyord=args.poly_detrend)
            # therefore, we do:
            # detrending is enabled
            if args.poly_detrend:
                poly_detrend(fds, polyord=args.poly_detrend, chunks_attr='chnks')
                # Event-related Pre-processing Is Not Event-related
                # some preprocessing is only meaningful when performed on the
                # full time series and not on the segmented event samples. An
                # example is detrending that typically needs to be done on the
                # original, continuous time series
            # no detrending
            else:
                pass

            # normalization is enabled
            if args.zscore:
                zscore(fds, chunks_attr='chnks', param_est=('trgts', args.zscore))
            # no normalization
            else:
                pass

            # simple average-sample approach is limited to block-design with a clear
            # temporal separation of all signals of interest, whereas HRF modeling is more suitable
            # for experiments with fast stimulation alternation
            evds = fit_event_hrf_model(fds,
                                       events,  # it is perfectly fine to have events that are not synchronized with
                                       # the TR. The labeling of events is taken from the 'events' list. Any
                                       # attribute(s) that are also in the dicts will be assigned as
                                       # condition labels in the output dataset
                                       time_attr='time_coords',  # identifies at which timepoints each BOLD volume was
                                       # acquired
                                       condition_attr=cond_attr  # name of the event attribute with the condition
                                       # labels. Can be a list of those (e.g. ['targets', 'chunks']) combination
                                       # of which would constitute a condition
                                       )

            # this function behaves identical to ZScoreMapper
            # the only difference is that the actual Z-scoring is done in-place
            # potentially causing a significant reduction of memory demands
            zscore(evds, chunks_attr=None)  # normalizes each feature (GLM parameters estimates for each voxel at
                                            # this point)

            ####################From Timeseries To Spatio-temporal Samples:####################

            # remember, each feature is now voxel-at-time-point, so we get a chance of looking at the spatio-temporal
            # profile of classification-relevant information in the data

            sens = cv_sensana(evds)
            sens_comb = sens.get_mapped(maxofabs_sample())  # another way to combine the sensitivity maps -> into a
                                                            # single map. It should be noted that sensitivities can
                                                            # not be directly compared to each other, even if they
                                                            # stem from the same algorithm and are just computed on
                                                            # different dataset splits. In an analysis one would have
                                                            # to normalize them first. PyMVPA offers, for example,
                                                            # l1_normed() and l2_normed() that can be used in
                                                            # conjunction with FxMapper to do that as a post-processing
                                                            # step
            nimg = map2nifti(fds, sens_comb)
            nimg.to_filename(os.path.join(args.output_dir, subj_name,
                                          subj_name + '_task-' + args.task + '_' + ROI_name +
                                          '_' + '_'.join(args.conditions_to_classify) + '_pattern.nii.gz'))

            html_str = """
            <!DOCTYPE html>
            <html>
            <body>
                <h2>ROI: %s</h2>
                <hr>
                <pre>%s</pre>
            </body>
            </html>
            """
            html_str = html_str % (ROI_name, cv_sensana.clf.ca.stats.as_string(description=True))
            subj_html.write(html_str)


        subj_html.close()

run_script('chmod -R 777 %s' % args.output_dir)
