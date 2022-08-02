#!/usr/bin/env python2.7
# for html searchlight visualization:
import matplotlib
matplotlib.use('Agg') # must be done before importing matplotlib.pyplot
import matplotlib.pyplot as pl
import base64
from io import BytesIO

import argparse
import csv
from glob import glob
import json
import mvpa2
# replace the following line with individual imports
from mvpa2.suite import *
import numpy as np
import os
from os import listdir
from os.path import join
import subprocess
import tempfile
import time

#import pylab as pl # this was named the same as pl - not sure why we need pylab
from mvpa2 import cfg

import mvpa2.datasets as md

# added for LSS implementation:
from copy import deepcopy

__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'version')).read()


def run_script(script):
    with tempfile.NamedTemporaryFile() as scriptfile:
        scriptfile.write(script)
        scriptfile.flush()
        subprocess.call(['/bin/bash', scriptfile.name])


parser = argparse.ArgumentParser(description='PyMVPA BIDS-App')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                                     'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                                       'should be stored.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                                           'Multiple participant level analyses can be run independently '
                                           '(in parallel) using the same output_dir.',
                    choices=['participant_prep', 'participant_test'])
parser.add_argument('-s', '--searchlight', help='Performs a spheric searchlight analysis with s being the '
                                                'radius of the spheres. If this parameter is not provided, '
                                                'ROI-based analysis will be run. (default: 3)',
                    nargs='?', const=3, type=int)
parser.add_argument('-k', '--task', help='Task to analyze. This has to be specified for both '
                                         'participant_prep and participant_test analysis levels.')
parser.add_argument('-c', '--conditions_to_classify', help='Conditions to classify.',
                    nargs="+")
parser.add_argument('-p', '--participant_label',
                    help='The label(s) of the participant(s) that should be analyzed. The label '
                         'corresponds to sub-<participant_label> from the BIDS spec '
                         '(so it does not include "sub-"). If this parameter is not '
                         'provided all subjects should be analyzed. Multiple '
                         'participants can be specified with a space separated list.',
                    nargs="+")
parser.add_argument('-l', '--noinfolabel', help='When building a sample attribute array from an event list, this will '
                                                'be the condition label to assign to all samples for which '
                                                'no stimulation condition information is contained in the events. '
                                                'For more information, look into PyMVPA\'s events2sample_attr. If '
                                                'this parameter is not provided no such labeling will be '
                                                'performed. (default: \'rest\')',
                    nargs='?', const='rest')
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
parser.add_argument('-m', '--lss', help='Enabling this parameter will run HRF modeling in an LSS fashion. '
                                        'Default is OLS.',
                    action='store_true')
parser.add_argument('-r', '--rsa', help='Enabling this parameter will run Representational Similarity Analysis. '
                                        'Default is classification.',
                    action='store_true')
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
        subjects=$path/derivatives/sub-$subjects
        [ -d "${subjects}" ] || continue
        subj_name=$(basename $subjects)
        mkdir $out_path/$subj_name
        fslmerge -t "$out_path/$subj_name/""$subj_name""_task-""$task""_bold_space_preproc" "$subjects/ses-01/func/"*"$task""_"*"-preproc_bold.nii.gz"
        #reading dim4 values of all runs
        for runs in "$path/derivatives/$subj_name/ses-01/func/"*"$task""_"*"-preproc_bold.nii.gz"; do
            fslval $runs dim4 >> "$out_path/$subj_name/""$subj_name""_task-""$task""_dim4.txt"
        done
    done
    '''
    my_bash_script = my_bash_script % (args.bids_dir, args.output_dir, args.task, ' '.join(subjects_to_analyze))
    run_script(my_bash_script)

# running participant_test level
elif args.analysis_level == "participant_test":
    if args.rsa:
        pl.figure(figsize=[10, 10]) # took out of plot_mtx
        figure, axis = pl.subplots(5, 6, figsize=(10,10))  # 5*6 = 30 subjects # took out of plot_mtx
        figure.tight_layout()
    group_html = open(os.path.join(args.output_dir, 'group.html'), 'w') # took it out of the above if

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
        for filename in sorted(os.listdir(os.path.join(args.bids_dir, subj_name, 'ses-01', 'func'))):
            keyname = args.task + '_'
            if (keyname in filename) and filename.endswith(".tsv"):
                with open(os.path.join(args.bids_dir, subj_name, 'ses-01', 'func', filename)) as tsvfile:
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

        if args.rsa:
            # ADDED FOR RSA TO GET ALL STIMULI LABELS:
            stim_order = []
            # with open(os.path.join(args.output_dir, 'stimorder', 'stimorder_' + subj_name + '.tsv')) as stimtsv:
            with open(os.path.join(args.output_dir, 'stimorder.tsv')) as stimtsv:
                reader = csv.DictReader(stimtsv, dialect='excel-tab')
                for row in reader:
                    stim_order.append(row['x'])
            # EVERY RSA WILL BE SORTED BASED ON stim_order

        events = [ev for ev in original_events]

        # events and cond_attr will later be passed to fit_event_hrf_model

        if args.condition_attr_onset:
            cond_attr = ('onset', 'targets', 'chunks')
            est = 'Individual Event' # will be used in the html output
        else:
            cond_attr = ('targets', 'chunks')
            est = 'Condition per each Run'

        clf = SVM() # SVMs come with sensitivity analyzers!
        # clf = LinearNuSVMC()

        if args.nfold_partitioner >= 1:
            cv_type = int(args.nfold_partitioner)
        else:
            cv_type = args.nfold_partitioner

        if args.rsa:
            # REPRESENTATIONAL SIMILARITY ANALYSIS
            fs = 'N/A'  # MIGHT REMOVE FOR RSA

            # WE NEED THE FOLLOWING HELPER FUNCTIONS TO PLOT DISSIMILARITY MATRICES
            # USING CORRELATION-DISTANCE -> COLORBAR RANGE OF [0,2]

            def plot_mtx_ind(mtx, labels, title):
                pl.figure(figsize=[10, 10])
                pl.imshow(mtx, interpolation='nearest')
                pl.xticks(range(len(mtx)), labels, rotation=-45, fontsize=5)
                pl.yticks(range(len(mtx)), labels, fontsize=5)
                pl.title(title)
                #pl.clim((0, 2)) # keeping this because keeping correlation-distance
                pl.clim((0, 1)) # where most colors are
                # pl.clim((-1, 1))
                pl.colorbar()


            def plot_mtx_grp(rw, clmn, mtx, labels, title):
                # replacing pl with axis[rw, clmn]
                axis[rw, clmn].imshow(mtx, interpolation='nearest')
                # pl.xticks(range(len(mtx)), labels, rotation=-45, fontsize=5)
                # pl.yticks(range(len(mtx)), labels, fontsize=5)
                # pl.title(title)
                axis[rw, clmn].title.set_text(title)
                axis[rw, clmn].set_xticks([])
                axis[rw, clmn].set_yticks([])
                # axis[rw, clmn].clim((0, 2)) # keeping this because keeping correlation-distance
                # axis[rw, clmn].colorbar()


            # ROI-based:
            if not args.searchlight:
                print 'Specific ROI-based settings of RSA'
            # Searchlight:
            else:
                print 'Specific searchlight settings of RSA'
        else:
            # CLASSIFICATION
            # ROI-based:
            if not args.searchlight:
                # feature selection is enabled -> note: our current feature selection is ANOVA-based and therefore univariate
                if args.feature_selection:
                    if args.feature_selection > 1:
                        fs = 'On (selected %d features)' % args.feature_selection
                        fsel = SensitivityBasedFeatureSelection(
                            OneWayAnova(),
                            FixedNElementTailSelector(int(args.feature_selection), mode='select', tail='upper')
                        )
                    elif args.feature_selection <= 1:
                        fs = 'On (selected %.2f%% of features)' % (args.feature_selection * 100)
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
            # Searchlight:
            else:
                fs = 'N/A'  # KEEPING FROM MVPA - MIGHT REMOVE FOR RSA

                cv = CrossValidation(clf, NFoldPartitioner(cvtype=cv_type))

                ##########PERMUTATION TESTING##########
                # # What does the classifier have to say about the actual data,
                # # but when it was "trained" on randomly permuted data:
                # partitioner = NFoldPartitioner()
                # repeater = Repeater(count=3)
                # # only once and only for samples that were labeled as being part of the training set in a particular CV-fold:
                # permutator = AttributePermutator('targets', limit={'partitions': 1}, count=1)
                # null_cv = CrossValidation(
                #     clf,
                #     # will cause the CV to permute the training set for each CV-fold internally:
                #     ChainNode([partitioner, permutator], space=partitioner.get_space()),
                #     errorfx=mean_mismatch_error)
                # distr_est = MCNullDist(repeater, tail='left', measure=null_cv,
                #                        enable_ca=['dist_samples'])
                # # cross-validation measure for computing the empricial performance estimate:
                # cv = CrossValidation(clf, partitioner, errorfx=mean_mismatch_error,
                #                      null_dist=distr_est, enable_ca=['stats'])
                #######################################


                plot_args = {
                    'background' : os.path.join(args.bids_dir, 'derivatives/' + subj_name + '/ses-01/anat/' + subj_name + '_ses-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'),
                    'background_mask' : os.path.join(args.bids_dir, 'derivatives/' + subj_name + '/ses-01/anat/' + subj_name + '_ses-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'),
                    'do_stretch_colors' : False,
                    'cmap_bg': 'gray',
                    'cmap_overlay': 'autumn',
                    'interactive' : True,
                }

        # draw2html is used for both RSA and Classification:
        def draw2html(group):
            tmpfile = BytesIO()
            pl.savefig(tmpfile, format='png')
            # pl.savefig('ROI.png', format='png')
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            html_pct = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
            if group == 0:
                subj_html.write(html_pct)
            elif group == 1:
                group_html.write(html_pct)

        subj_html = open(os.path.join(args.output_dir, subj_name + '.html'), 'w')

        # NEED TO MAKE CHANGES TO THE FOLLOWING FOR RSA:
        html_str = """
        <!DOCTYPE html>
        <html>
        <body>
                <h1 style="font-family:amatic;">Summary</h1>
                <ul>
                <li>Subject ID: %s</li>
                <li>Functional Series: %s</li>
                <li>Task: %s</li>
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
        html_str = html_str % (subj_name, number_of_runs, args.task, args.noinfolabel,
                               detr, norm, est, fs, cv_type)
        subj_html.write(html_str)

        # ROI-based (RSA & Classification):
        if not args.searchlight:
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
                    zscore(fds, chunks_attr='chnks')
                    #zscore(fds, chunks_attr='chnks', param_est=('trgts', args.zscore))
                # no normalization
                else:
                    pass

                # simple average-sample approach is limited to block-design with a clear
                # temporal separation of all signals of interest, whereas HRF modeling is more suitable
                # for experiments with fast stimulation alternation

                print '####################'
                print 'HRF MODELING STARTED'
                print '####################'
                if not args.lss:
                    print 'OLS is running...'
                    evds = fit_event_hrf_model(fds,
                                               events, # it is perfectly fine to have events that are not synchronized with
                                               # the TR. The labeling of events is taken from the 'events' list. Any
                                               # attribute(s) that are also in the dicts will be assigned as
                                               # condition labels in the output dataset

                                               # note that original_events and events are the same thing, each with 225 events
                                               # they contain events in the right time order, as is available in stimorder files too
                                               time_attr='time_coords', # identifies at which timepoints each BOLD volume was acquired
                                               condition_attr=cond_attr # name of the event attribute with the condition
                                               # labels. Can be a list of those (e.g. ['targets', 'chunks']) combination
                                               # of which would constitute a condition

                                               # glmfit_kwargs = dict(model='ar1') # old by default
                                               # regr_attrs=['gsr'] #removed for the Tone Scramble analysis
                                               )
                else:
                    # need to write code for RSA (because of individual trial names in tsv files) and then
                    # separately for classification
                    print 'LSS is running...'
                    # Note: '=' and 'copy()' make shallow copies of dictionaries! For deep copies use deepcopy
                    # Pay attention to APPEND as well
                    # **********CHECK ALL PREVIOUS CODES WITH DICTIONARIES**********
                    # the following was the first hard-coded approach:

                    curr_evnts = []
                    cntr = 0
                    for ev in events:
                        evcopy = deepcopy(ev)
                        curr_evnts.append(evcopy)
                        if evcopy['targets'].split('_')[0] == 'HH':
                            curr_evnts[cntr]['targets'] = 'social'
                        elif evcopy['targets'].split('_')[0] == 'OO':
                            curr_evnts[cntr]['targets'] = 'physical'
                        cntr = cntr + 1

                    for trial in range(0, len(events)):
                        # changing only the current trial's label:

                        # old way of copying events onto curr_evnts:

                        eventscopy = deepcopy(events[trial])
                        store = deepcopy(curr_evnts[trial])
                        curr_evnts[trial]['targets'] = eventscopy['targets']

                        # store = events[trial]['targets']
                        # events[trial]['targets'] = 'currenttrial'
                        single_evds = fit_event_hrf_model(fds,
                                                          # events,
                                                          curr_evnts,
                                                          time_attr='time_coords',
                                                          condition_attr=cond_attr
                                                          )
                        curr_evnts[trial]['targets'] = store['targets']
                        #events[trial]['targets'] = store
                        # Keep the single trial only:
                        single_evds = single_evds[np.array([l not in args.conditions_to_classify
                                                            for l in single_evds.targets], dtype='bool')]
                        if trial == 0:
                            evds = single_evds
                        else:
                            evds = md.vstack((evds, single_evds))
                print '####################'
                print 'HRF MODELING ENDED'
                print '####################'
                # shape of evds.shape is (225, 1496) -> 225 trials and 1496 voxels in the rTPJ

                if args.rsa:
                    # NOTE THAT ORDER OF STIMULI GETS MIXED FROM THIS POINT
                    # LET'S SORT IT THEN USING stim_order
                    print 'SORTING TRIALS...'
                    counter = 0
                    for ord in stim_order:
                        for srch in evds: # evds might also change and we might need deepcopy on evds
                            if srch.sa.targets == ord:
                                if counter == 0:
                                    sorted_evds = srch
                                    # another way: ds1=md.Dataset([5,6,7])
                                    # md.vstack((ds1,ds2))
                                else:
                                    sorted_evds = md.vstack((sorted_evds, srch))
                                    # sorted_evds = sorted_evds + srch # unsupported operand type(s) for +: 'Dataset' and 'Dataset'
                                # sorted_evds[counter] = srch
                                # sorted_evds[counter] = srch.copy(deep=True, sa=None, fa=None, a=None)
                                counter = counter + 1
                                break
                    # evds = sorted_evds
                else:
                    pass

                # this function behaves identical to ZScoreMapper
                # the only difference is that the actual Z-scoring is done in-place
                # potentially causing a significant reduction of memory demands
                if args.zscore: # normalizes each feature (GLM parameters estimates for each voxel at this point)
                    if args.rsa: # will rename sorted_evds to evds
                        zscore(sorted_evds, chunks_attr='chunks')
                    else:
                        zscore(evds, chunks_attr='chunks')
                # no normalization
                else:
                    pass

                if args.rsa:
                    print 'ROI-based RSA is running...'
                    # MIGHT NEVER USE mtds
                    mtgs = mean_group_sample(['targets'])
                    mtds = sorted_evds.get_mapped(mtgs)

                    mtcgs = mean_group_sample(['targets', 'chunks'])
                    mtcds = sorted_evds.get_mapped(mtcgs)

                    # WE HAVE evds RATHER THAN mtds FROM THE TUTORIAL

                    from mvpa2.measures import rsa

                    # DISSIMILARITY MATRIX FOR THE ENTIRE ROI:
                    # CORRELATION DISTANCE:
                    dsm = rsa.PDist(square=True)
                    RDM_title = 'Correlation'
                    # EUCLIDEAN:
                    # dsm = rsa.PDist(square=True, pairwise_metric='euclidean')
                    # RDM_title = 'Euclidean'
                    # MAHALANOBIS:
                    # dsm = rsa.PDist(square=True, pairwise_metric='mahalanobis')
                    # RDM_title = 'Mahalanobis'

                    res = dsm(sorted_evds)
                    curr_mat = np.matrix(res.samples)
                    # curr_mat = 1 - curr_mat  # changes distance to correlation r = 1-d
                    if subj_name == 'sub-01':
                        ave_res = curr_mat
                    else:
                        ave_res = ave_res + curr_mat
                    # printing res would give <Dataset: 225x225@float64, <sa: chunks,onset,regressors,targets>>
                    # plot_mtx(res, sorted_evds.sa.targets, 'ROI Pattern Correlation Distances')
                    if subj_name == 'sub-01' or subj_name == 'sub-02' or subj_name == 'sub-03' or subj_name == 'sub-04' or subj_name == 'sub-05' or subj_name == 'sub-06':
                        row = 0
                    elif subj_name == 'sub-07' or subj_name == 'sub-08' or subj_name == 'sub-09' or subj_name == 'sub-10' or subj_name == 'sub-11' or subj_name == 'sub-12':
                        row = 1
                    elif subj_name == 'sub-13' or subj_name == 'sub-14' or subj_name == 'sub-15' or subj_name == 'sub-16' or subj_name == 'sub-17' or subj_name == 'sub-18':
                        row = 2
                    elif subj_name == 'sub-19' or subj_name == 'sub-20' or subj_name == 'sub-21' or subj_name == 'sub-22' or subj_name == 'sub-23' or subj_name == 'sub-24':
                        row = 3
                    elif subj_name == 'sub-25' or subj_name == 'sub-26' or subj_name == 'sub-27' or subj_name == 'sub-28' or subj_name == 'sub-29' or subj_name == 'sub-30':
                        row = 4

                    if subj_name == 'sub-01' or subj_name == 'sub-07' or subj_name == 'sub-13' or subj_name == 'sub-19' or subj_name == 'sub-25':
                        column = 0
                    elif subj_name == 'sub-02' or subj_name == 'sub-08' or subj_name == 'sub-14' or subj_name == 'sub-20' or subj_name == 'sub-26':
                        column = 1
                    elif subj_name == 'sub-03' or subj_name == 'sub-09' or subj_name == 'sub-15' or subj_name == 'sub-21' or subj_name == 'sub-27':
                        column = 2
                    elif subj_name == 'sub-04' or subj_name == 'sub-10' or subj_name == 'sub-16' or subj_name == 'sub-22' or subj_name == 'sub-28':
                        column = 3
                    elif subj_name == 'sub-05' or subj_name == 'sub-11' or subj_name == 'sub-17' or subj_name == 'sub-23' or subj_name == 'sub-29':
                        column = 4
                    elif subj_name == 'sub-06' or subj_name == 'sub-12' or subj_name == 'sub-18' or subj_name == 'sub-24' or subj_name == 'sub-30':
                        column = 5

                    #plot_mtx_ind(curr_mat, res.sa.targets, subj_name + ' ' + ROI_name + ' Pattern ' + RDM_title + ' Distances')
                    #draw2html(0)

                    plot_mtx_grp(row, column, curr_mat, res.sa.targets, subj_name)
                    draw2html(1)

                    if subj_name == 'sub-30':
                        ave_res = ave_res / 30
                        plot_mtx_ind(ave_res, res.sa.targets, 'Average ROI Pattern Correlation Distances')
                        draw2html(1)
                else:
                    print 'ROI-based Classification is running...'
                    # average betas (added recently for studyforrest emotions): NOT IMPLEMENTED YET
                    # evds = evds.get_mapped(mean_group_sample(['targets', 'chunks']))

                    ####################From Timeseries To Spatio-temporal Samples:####################

                    # remember, each feature is now voxel-at-time-point, so we get a chance of looking at the spatio-temporal
                    # profile of classification-relevant information in the data

                    # NEED TO REMOVE CONDITIONS FROM evds BASED ON CONDITIONS-TO-CLASSIFY? Look at Searchlight below

                    sens = cv_sensana(evds)
                    sens_comb = sens.get_mapped(
                        maxofabs_sample())  # another way to combine the sensitivity maps -> into a
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
                                                  '_' + 'all' + '_pattern.nii.gz'))

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
        # Searchlight (RSA & Classification):
        else:
            ROIs = []
            # need to have one single mask for searchlight - if none, use whole brain
            ROIs = sorted(os.listdir(os.path.join(args.output_dir, 'masks')))
            ROI_name = ROIs[0].split('.')[0]
            mask_fname = os.path.join(args.output_dir, 'masks', ROIs[0])
            fds = fmri_dataset(samples=all_runs_bold_fname,
                               mask=mask_fname)
            # fds = fmri_dataset(samples=all_runs_bold_fname)
            # for extracting GSR:
            ##########Disabling for the Tone Scramble project##########
            # print("SUBJECT NAME:")
            # print(subj_name)
            # mask_fname = os.path.join(args.output_dir, 'masks',
            #                          subj_name + '_wm.nii')
            # fds_wm = fmri_dataset(samples=all_runs_bold_fname,
            #                      mask=mask_fname)
            # global_signal = []
            # for smpl in range(0, fds_wm.shape[0]):
            #    global_signal.append(np.mean(fds_wm.samples[smpl,:]))
            # fds.sa['gsr'] = global_signal

            fds.sa['chnks'] = chunks_labels
            targets_labels = events2sample_attr(original_events, fds.sa.time_coords, noinfolabel=args.noinfolabel,
                                                condition_attr='targets')
            fds.sa['trgts'] = targets_labels

            # detrending is enabled
            if args.poly_detrend:
                poly_detrend(fds, polyord=args.poly_detrend, chunks_attr='chnks')
            # no detrending
            else:
                pass

            # normalization is enabled
            if args.zscore:
                # changed to param_est to None so that all samples will be used for parameter estimation
                zscore(fds, chunks_attr='chnks')
                # zscore(fds, chunks_attr='chnks', param_est=('trgts', args.zscore))
            # no normalization
            else:
                pass
            # remove here?!:
            # fds = fds[np.array([l in args.conditions_to_classify
            #                            for l in fds.sa.trgts], dtype='bool')]

            # shape of fds.shape is (1155, 1496)
            # rows are samples (for fds, those are volumes) - columns are features (for fds, those are voxels)

            print '####################'
            print 'HRF MODELING STARTED'
            print '####################'
            if not args.lss:
                print 'OLS is running...'
                evds = fit_event_hrf_model(fds,
                                           events,
                                           # note that original_events and events are the same thing, each with 225 events
                                           # they contain events in the right time order, as is available in stimorder files too
                                           time_attr='time_coords',
                                           condition_attr=cond_attr
                                           # glmfit_kwargs = dict(model='ar1') # old by default
                                           # regr_attrs=['gsr'] #removed for the Tone Scramble analysis
                                           )
            else:
                # for RSA we have individual trial names and need to code accordingly
                print 'LSS is running...'
                for trial in range(0, len(events)):
                    # changing only the current trial's label:
                    store = deepcopy(events[trial])
                    events[trial]['targets'] = 'currenttrial'
                    single_evds = fit_event_hrf_model(fds,
                                                      events,
                                                      time_attr='time_coords',
                                                      condition_attr=cond_attr
                                                      )
                    # Keep the single trial only:
                    single_evds = single_evds[np.array([l in ['currenttrial']
                                                        for l in single_evds.targets], dtype='bool')]
                    # at this point, single_evds is a 1 by 6713 Dataset
                    single_evds.sa['targets'] = [store['targets']]
                    if trial == 0:
                        evds = single_evds
                    else:
                        evds = md.vstack((evds, single_evds))
                    events[trial]['targets'] = store['targets']
            print '####################'
            print 'HRF MODELING ENDED'
            print '####################'

            if args.rsa:
                print 'SORTING...'
                counter = 0
                for ord in stim_order:
                    for srch in evds:
                        if srch.sa.targets == ord:
                            if counter == 0:
                                sorted_evds = srch
                                # another way: ds1=md.Dataset([5,6,7])
                                # md.vstack((ds1,ds2))
                            else:
                                sorted_evds = md.vstack((sorted_evds, srch))
                                # sorted_evds = sorted_evds + srch # unsupported operand type(s) for +: 'Dataset' and 'Dataset'
                            # sorted_evds[counter] = srch
                            # sorted_evds[counter] = srch.copy(deep=True, sa=None, fa=None, a=None)
                            counter = counter + 1
                            break
            else:
                pass

            # Inclusion of the additional regressor will not alter the beta-estimate of the hrf predictor,
            # but simply remove variance from the residual, which in turn will improve the statistics
            if args.zscore:
                if args.rsa:
                    zscore(sorted_evds, chunks_attr='chunks')
                else:
                    zscore(evds, chunks_attr='chunks')
            # no normalization
            else:
                pass

            # average betas: NOT IMPLEMENTED YET
            # evds.sa = evds.get_mapped(mean_group_sample(['targets', 'chunks']))

            if args.rsa:
                print 'Searchlight RSA is running...'
                '''
                # DISSIMILARITY IN A SEARCHLIGHT FASHION:
                from mvpa2.measures.searchlight import sphere_searchlight

                dsm = rsa.PDist(square=False)
                sl = sphere_searchlight(dsm, 2, nproc=15)
                print 'RDM SEARCHLIGHT STARTED'
                slres = sl(evds)
                print 'RDM SEARCHLIGHT ENDED'

                # score each searchlight sphere result wrt global pattern dissimilarity:
                distinctiveness = np.sum(np.abs(slres), axis=0)
                print 'Most Dissimilar Pattern Around', \
                    evds.fa.voxel_indices[distinctiveness.argmax()]
                # looking into this dissimilarity structure:
                from scipy.spatial.distance import squareform

                plot_mtx(squareform(slres.samples[:, distinctiveness.argmax()]),
                         evds.sa.targets,
                         'Max Distinctive Searchlight Pattern Correlation Distances')

                # html visualization:
                draw2html()
                '''

                '''
                # HOW CORRELATED ARE THE STRUCTURES ACROSS RUNS
                dscm = rsa.PDistConsistency()
                sl_cons = sphere_searchlight(dscm, 2)
                slres_cons = sl_cons(evds)

                mean_consistency = np.mean(slres_cons, axis=0)
                print 'Most Stable Dissimilarity Patterns Around', \
                    evds.fa.voxel_indices[mean_consistency.argmax()]
                plot_mtx(squareform(slres.samples[:, mean_consistency.argmax()]),
                         evds.sa.targets,
                         'Most Consistent Searchlight Pattern Correlation Distances')

                draw2html()

                # WHERE DO WE FIND DISSIMILARITY STRUCTURES THAT ARE SIMILAR TO THE MOST STABLE ONE
                tdsm = rsa.PDistTargetSimilarity(
                    slres.samples[:, mean_consistency.argmax()])
                from mvpa2.base.learner import ChainLearner
                from mvpa2.mappers.shape import TransposeMapper

                sl_tdsm = sphere_searchlight(ChainLearner([tdsm, TransposeMapper()]), 2)
                slres_tdsm = sl_tdsm(evds)

                # MAP BACK TO 3D VOXEL GRID, OVERLAY ONTO ANATOMY
                niftiresults = map2nifti(fds, slres_tdsm)
                niftiresults.to_filename(os.path.join(args.output_dir, subj_name,
                                                      subj_name + '_task-' + args.task + '_searchlight-' + str(
                                                          args.searchlight) +
                                                      '_similarity-' + 'all' + '.nii.gz'))
                '''

                '''
                # WHERE DO WE FIND DISSIMILARITY STRUCTURES THAT ARE SIMILAR TO THE COMPUTATIONAL MODEL
                import pandas as pd

                comp_mtrx = pd.read_csv(os.path.join(args.output_dir, 'compmtrx.tsv'), sep="\t", header=None)

                tdsm = rsa.PDistTargetSimilarity(
                    comp_mtrx)
                from mvpa2.base.learner import ChainLearner
                from mvpa2.mappers.shape import TransposeMapper

                print 'SIMILARITY SEARCHLIGHT STARTED'
                sl_tdsm = sphere_searchlight(ChainLearner([tdsm, TransposeMapper()]), radius=2, nproc=15)
                slres_tdsm = sl_tdsm(sorted_evds)

                # MAP BACK TO 3D VOXEL GRID, OVERLAY ONTO ANATOMY
                niftiresults = map2nifti(fds, slres_tdsm)
                niftiresults.to_filename(os.path.join(args.output_dir, subj_name,
                                                      subj_name + '_task-' + args.task + '_searchlight-' + str(
                                                          args.searchlight) +
                                                      '_similarity-' + 'all' + '.nii.gz'))
                '''
            else:
                print 'Searchlight Classification is running...'

                # last thing added for test:
                # center_ids = evds.fa.nonzero()[0]

                # determines local neighborhoods -> space='voxel_indices'

                evds = evds[np.array([l in args.conditions_to_classify
                                      for l in evds.targets], dtype='bool')]

                sl = sphere_searchlight(cv, radius=args.searchlight, space='voxel_indices', nproc=15,
                                        postproc=mean_sample())
                print("Searchlight Started!")
                start_time = time.time()
                res = sl(evds) # why was this commented?
                print("--- %s seconds ---" % (time.time() - start_time))
                print("Searchlight Ended!")

                # transforming error maps into accuracies:
                res.samples *= -1
                res.samples += 1

                niftiresults = map2nifti(fds, res)
                niftiresults.to_filename(os.path.join(args.output_dir, subj_name,
                                                      subj_name + '_task-' + args.task + '_searchlight-' + str(
                                                          args.searchlight) +
                                                      '_pattern-' + '-'.join(args.conditions_to_classify) + '.nii.gz'))

                # Here goes plotting figures:
                fig = pl.figure(figsize=(60, 60), facecolor='white')
                subfig = plot_lightbox(overlay=niftiresults,
                                       vlim=(None, None),
                                       fig=fig, **plot_args)
                pl.title('Accuracy Distribution for Radius %i' % args.searchlight)

                sphere_errors = res.samples[0]
                res_mean = np.mean(res)
                res_std = np.std(res)
                # hard-coded!
                chance_level = 0.5  # 1.0 - (1.0 / len(ds.uniquetargets))
                # for how many spheres the error is more the two standard deviations lower than chance:
                frac_lower = np.round(np.mean(sphere_errors < chance_level - 2 * res_std), 3)
                print(frac_lower)

        subj_html.close()
    group_html.close()

run_script('chmod -R 777 %s' % args.output_dir)
