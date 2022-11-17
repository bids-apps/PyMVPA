#!/usr/bin/env python2.7
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import base64
from io import BytesIO

import argparse
import csv
from glob import glob
import json
import mvpa2
from mvpa2.suite import *
import numpy as np
import os
from os import listdir
from os.path import join
import subprocess
import tempfile
import time
from mvpa2.clfs.svm import LinearCSVMC

# added to load surfaces:
from mvpa2.support.nibabel.surf import read as surfread # general read function for surfaces
from mvpa2.datasets.gifti import gifti_dataset as giiread
from mvpa2.support.nibabel.surf_gifti import read as surfgiiread

from mvpa2 import cfg
import mvpa2.datasets as md
import nibabel as nib
from copy import deepcopy

from mvpa2.measures import rsa

__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'version')).read()


def run_script(script):
    with tempfile.NamedTemporaryFile() as scriptfile:
        scriptfile.write(script)
        scriptfile.flush()
        subprocess.call(['/bin/bash', scriptfile.name])


parser = argparse.ArgumentParser(description='PyMVPA BIDS-App')
parser.add_argument('bids_dir', help='The input directory with dataset '
                                     'formatted according to BIDS standard.')
parser.add_argument('output_dir', help='The directory that will store '
                                       'outputs.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                                           'Multiple analyses can run independently '
                                           'in parallel.',
                    choices=['participant_prep', 'participant_test'])
parser.add_argument('-p', '--participant_id',
                    help='Subjects (specified by ID) that will be analyzed. The ID '
                         'corresponds to sub-<participant_id> from BIDS specification. '
                         'If specific IDs are not provided, all subjects will be '
                         'analyzed. Multiple subjects can be specified by a space '
                         'separated list.',
                    nargs="+")
parser.add_argument('-s', '--session',
                    help='Session ID for multi-session datasets.',
                    nargs='?')
parser.add_argument('--searchlight', help='Performs searchlight analysis with s being the radius of '
                                                'spheres/discs in volumetric/surface mode. If this flag is '
                                                'not enabled, ROI-based analysis will run. (default: 3.0)',
                    nargs='?', const=3.0, type=float)
parser.add_argument('-t', '--task', help='Task to analyze. This has to be specified for both '
                                         'participant_prep and participant_test analysis levels.')
parser.add_argument('-c', '--conditions_to_classify', help='Conditions to classify.',
                    nargs="+")
parser.add_argument('--noinfolabel', help='When building a sample attribute array from an event list, this will '
                                                'be assigned to all samples with no label. For more information, check '
                                                'PyMVPA\'s events2sample_attr. (default: \'rest\')',
                    nargs='?', const='rest')
parser.add_argument('--poly_detrend', help='Order of Legendre polynomial to remove from the data. This '
                                                 'will remove every polynomial up to and including the provided '
                                                 'value. If this parameter is not provided no detrending will '
                                                 'be performed. (default: 1)',
                    nargs='?', const=1, type=int)
parser.add_argument('--zscore', help='Feature-wise, run-wise z-scoring of time-series. Scales '
                                           'all features into approximately the same range, and removes '
                                           'their mean. If this parameter is not provided no normalization '
                                           'will be performed.',
                    nargs='?')
parser.add_argument('-i', '--indiv_trials', help='When (HRF) modeling the time-series, enabling this flag '
                                                         'will estimate betas per individual trials, rather than per '
                                                         'condition per run. This provides more but noisier estimates. '
                                                         'A trade off between the number of training samples and noise '
                                                         'reduction should be made by the user.',
                    action='store_true')
parser.add_argument('-f', '--feature_select', help='Uses ANOVA to select features with highest F-scores. '
                                                      'Will perform FixedNElementTailSelector if f > 1, and '
                                                      'FractionTailSelector if f < 1. If this parameter is not provided '
                                                      'no feature selection will be performed.',
                    type=float)
parser.add_argument('--cvtype', help='When running cross-validation on a dataset with n runs, with '
                                                      'cvtype = 1 (default), n partition sets will be generated, where '
                                                      'each run is sequentially left out to form one partition, with all '
                                                      'other runs forming the other partition together. If cvtype > 1, '
                                                      'all possible combinations of cvtype number of runs are left out.',
                    default=1, type=float)
parser.add_argument('--lss', help='Enabling this parameter will run GLM in a Least Squares Single (LSS) '
                                        'fashion. Default is Ordinary Least Squares (OLS).',
                    action='store_true')
parser.add_argument('--rsa', help='Enabling this parameter will run Representational Similarity Analysis. '
                                        'Default is classification.',
                    action='store_true')
parser.add_argument('--surf', help='Enabling this parameter will run analyses on surface. '
                                        'Default is volumetric.',
                    action='store_true')
parser.add_argument('--space', help='Surface space of analysis. Options are fsnative or fsaverage.',
                    nargs='?', choices=['fsnative', 'fsaverage'], default='fsaverage')
parser.add_argument('--hemi', help='Hemisphere of analysis. Need to specify for surface. '
                    'Options are l (left) or r (right).',
                    nargs='?', choices=['l', 'r'], default='r')
parser.add_argument('--mask', help='NIfTI mask used for both ROI-based analysis '
                                         'and searchlight.')
parser.add_argument('--dist', help='Distance metric to use for Representational Dissimilarity '
                                                'Matrices (RDM). Choices are correlation (default), euclidean, '
                                                'and mahalanobis.',
                    nargs='?', choices=['correlation', 'euclidean', 'mahalanobis'], default='correlation')
parser.add_argument('--nproc', help='Number of CPUs to use (default: 8).',
                    nargs='?', const=8, type=int)
parser.add_argument('--skip_bids_validator', help='Skipping BIDS validation',
                    action='store_true')
parser.add_argument('-v', '--version', action='version',
                    version='PyMVPA BIDS-App Version {}'.format(__version__))

args = parser.parse_args()

if not args.skip_bids_validator:
    run_script('bids-validator %s' % args.bids_dir)

os.chdir(args.bids_dir)

subjects_to_analyze = []
# only for a subset of subjects
if args.participant_id:
    subjects_to_analyze = args.participant_id
# for all subjects
else:
    subject_dirs = sorted(glob(os.path.join(args.bids_dir, "sub-*")))
    subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]

if args.session:
    session_id = args.session
else:
    session_id = None

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
    sess=%s
    for subjects in ${sub_IDs}; do
        subjects=$path/derivatives/sub-$subjects
        [ -d "${subjects}" ] || continue
        subj_name=$(basename $subjects)
        mkdir $out_path/$subj_name
        if [ $sess = "None" ]
        then
            echo "No session"
            addr="func/"
        else
            echo "Multi-session"
            addr="ses-""$sess""/func/"
        fi
        fslmerge -t "$out_path/$subj_name/""$subj_name""_task-""$task""_desc-preproc_bold" "$subjects/$addr"*"$task"_*"-preproc_bold.nii.gz"
        #reading dim4 values of all runs
        for runs in "$path/derivatives/$subj_name/$addr"*"$task"_*"-preproc_bold.nii.gz"; do
            fslval $runs dim4 >> "$out_path/$subj_name/""$subj_name""_task-""$task""_dim4.txt"
        done
    done
    '''
    my_bash_script = my_bash_script % (args.bids_dir, args.output_dir, args.task, ' '.join(subjects_to_analyze), session_id)
    run_script(my_bash_script)

# running participant_test level
elif args.analysis_level == "participant_test":
    for subjects in subjects_to_analyze:
        subj_name = 'sub-' + str(subjects)

        chunks = []
        duration = []
        onset = []
        targets = []
        chunk_counter = 0

        if session_id == None:
            addr = os.path.join(args.bids_dir, subj_name, 'func')
        else:
            addr = os.path.join(args.bids_dir, subj_name, 'ses-' + session_id, 'func')
        # looping through all runs' tsv events files
        for filename in sorted(os.listdir(addr)):
            keyname = args.task + '_'
            if (keyname in filename) and filename.endswith(".tsv"):
                with open(os.path.join(addr, filename)) as tsvfile:
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
            # RDMs will be ordered based on stimorder.tsv
            stim_order = []
            with open(os.path.join(args.output_dir, 'stimorder.tsv')) as stimtsv:
                reader = csv.reader(stimtsv, delimiter="\t")
                for row in reader:
                    stim_order.append(row[0]) # each row is a list, with only one element in each list

        events = [ev for ev in original_events]

        # events and cond_attr will later be passed to fit_event_hrf_model

        if args.indiv_trials:
            cond_attr = ('onset', 'targets', 'chunks')
            est = 'Individual Trial' # will be used in html output
        else:
            cond_attr = ('targets', 'chunks')
            est = 'Condition per Run'

        # clf = SVM() # SVMs come with sensitivity analyzers
        clf = LinearCSVMC()

        if args.cvtype >= 1:
            cv_type = int(args.cvtype)
        else:
            cv_type = args.cvtype

        # Pipeline Tree:
        # Volume
        #           ROI-based
        #                   classification
        #                   RSA
        #           Searchlight
        #                   classification
        #                   RSA
        # Surface
        #           ROI-based
        #                   classification
        #                   RSA
        #           Searchlight
        #                   classification
        #                   RSA

        ###############################
        ########## ROI-based ##########
        ###############################
        if not args.searchlight:
            ##########################
            ##### Classification #####
            ##########################
            if not args.rsa:
                # Current feature selection is ANOVA-based and univariate - not recommended
                if args.feature_select:
                    if args.feature_select > 1:
                        fs = 'On (selected %d features)' % args.feature_select
                        fsel = SensitivityBasedFeatureSelection(
                            OneWayAnova(),
                            FixedNElementTailSelector(int(args.feature_select), mode='select', tail='upper')
                        )
                    elif args.feature_select <= 1:
                        fs = 'On (selected %.2f%% of features)' % (args.feature_select * 100)
                        fsel = SensitivityBasedFeatureSelection(
                            OneWayAnova(),
                            FractionTailSelector(args.feature_select, mode='select', tail='upper')
                        )
                    clf = FeatureSelectionClassifier(clf, fsel)
                else: # no feature selection
                    fs = 'Off'
                    pass

                # a convenient way to access the overall performance of the underlying classifier, and get sensitivities at
                # the same time: (can effectively perform a cross-validation analysis internally)
                sclf = SplitClassifier(clf, NFoldPartitioner(cvtype=cv_type),
                                       # exp: if cvtype=5 and the total number of runs is 12, then we'll
                                       # have C(12,5)=12!/((12-5)!*5!) ways of splitting the data, and
                                       # will have 5 runs for testing in each combination
                                       enable_ca=['stats'])
                cv_sensana = sclf.get_sensitivity_analyzer()  # no post-processing here -> obtaining sensitivity maps from all
                # internally trained classifiers =
                # C(number of conditions_to_classify,2)*number_of_runs maps
            ###############
            ##### RSA #####
            ###############
            else:
                print('ROI-based settings of RSA')
                fs = 'N/A'  # remove for RSA?

                def plot_mtx(mtx, labels, title):
                    pl.figure(figsize=[10, 10])
                    pl.imshow(mtx, interpolation='nearest')
                    pl.xticks(range(len(mtx)), labels, rotation=-45, fontsize=5)
                    pl.yticks(range(len(mtx)), labels, fontsize=5)
                    pl.title(title)
                    pl.clim((0, 2)) # correlation-distance
                    pl.colorbar()
        #################################
        ########## Searchlight ##########
        #################################
        else:
            ##########################
            ##### Classification #####
            ##########################
            if not args.rsa:
                fs = 'N/A'  # keeping for classification - remove for RSA?

                cv = CrossValidation(clf,
                                     NFoldPartitioner(cvtype=cv_type),
                                     errorfx=lambda p, t: np.mean(p == t), # so that accuracies are returned, not errors
                                     enable_ca=['stats'])
                '''
                plot_args = {
                    'do_stretch_colors': False,
                    'cmap_bg': 'gray',
                    'cmap_overlay': 'autumn',
                    'interactive': True,
                }
                '''
            ###############
            ##### RSA #####
            ###############
            else:
                print('Searchlight settings of RSA')

        # draw2html is used for both classification and RSA:
        def draw2html():
            tmpfile = BytesIO()
            pl.savefig(tmpfile, format='png')
            # pl.savefig('ROI.png', format='png')
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            html_pct = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
            subj_html.write(html_pct)

        subj_html = open(os.path.join(args.output_dir, subj_name + '.html'), 'w')

        html_str = """
        <!DOCTYPE html>
        <html>
        <body>
                <h1 style="font-family:amatic;">Summary</h1>
                <ul>
                <li>Subject ID: %s</li>
                <li># of Functional Runs: %s</li>
                <li>Task: %s</li>
                <li>No-Info Label: '%s'</li>
                <li>De-trending: %s</li>
                <li>Z-scoring: %s</li>
                <li>One Estimate per %s</li>
                <li>Feature Selection: %s</li>
                <li>NFold Partitioner: %s</li>
                </ul>
        </body>
        </html>
        """
        detr = 'On (polyord=%d)' % args.poly_detrend if args.poly_detrend else 'Off'
        norm = 'On' if args.zscore else 'Off'
        html_str = html_str % (subj_name, number_of_runs, args.task, args.noinfolabel,
                               detr, norm, est, fs, cv_type)
        subj_html.write(html_str)

        ########################################################################
        ############################## Loading... ##############################
        ########################################################################
        # reading time_coords info from volumetric data -> need to load both volume & surface:
        all_runs_bold_fname = os.path.join(args.output_dir, subj_name,
                                           subj_name + '_task-' + args.task + '_desc-preproc_bold.nii.gz')
        if args.mask == None: # no masking
            runallnii = fmri_dataset(samples=all_runs_bold_fname)
        else: # masking is enabled
            mask_fname = os.path.join(args.output_dir, 'masks', args.mask + '.nii')
            runallnii = fmri_dataset(samples=all_runs_bold_fname,
                               mask=mask_fname)

        # chunks_labels = events2sample_attr(original_events, fds.sa.time_coords, condition_attr='chunks')
        # rather than using events2sample_attr (or assign_conditionlabels) to attribute chunks labels to
        # samples which would be tricky because of noinfolabel, we do:
        runallnii.sa['chnks'] = chunks_labels  # we call this sample attribute 'chnks' so later it won't be mistaken for
        # 'chunks' in events
        targets_labels = events2sample_attr(original_events, runallnii.sa.time_coords, noinfolabel=args.noinfolabel,
                                            condition_attr='targets')
        runallnii.sa['trgts'] = targets_labels

        # find_events won't work for this app because we consider the possibility of jittering
        # events = find_events(targets=fds.sa.trgts, chunks=fds.sa.chnks)

        # note that we will be using 'chnks' and 'trgts' sample attributes for detrending and normalization
        # purposes, and not for cross-validation because with jittering, a lot of information
        # will be lost that way -> events will be used rather than samples

        print("NIfTI Dataset:")
        print(runallnii)

        if session_id == None:
            addr = os.path.join(args.bids_dir, 'derivatives', subj_name, 'func')
        else:
            addr = os.path.join(args.bids_dir, 'derivatives', subj_name, 'ses-' + session_id, 'func')
        if args.hemi == 'r':
            hemisphere = 'hemi-' + 'R'
        else:
            hemisphere = 'hemi-' + 'L'
        runallgii = []
        for filename in sorted(os.listdir(addr)):
            keyname = args.task + '_'
            if (keyname in filename) and filename.endswith(args.space + "_bold.func.gii") and (hemisphere in filename):
                currgii = giiread(os.path.join(addr, filename))
                runallgii.append(currgii)
        runallgii = md.vstack((runallgii))

        runallgii.sa['time_coords'] = runallnii.sa.time_coords
        # runallgii.fa['node_indices'] = np.arange(runallgii.shape[1], dtype=int)
        # Matteo Visconti's approach: (check if equal to the above)
        runallgii.fa['node_indices'] = np.arange(runallgii.nfeatures) # ONLY IF DATA NOT MASKED

        runallgii.sa['chnks'] = chunks_labels
        targets_labels = events2sample_attr(original_events, runallgii.sa.time_coords, noinfolabel=args.noinfolabel,
                                            condition_attr='targets')
        runallgii.sa['trgts'] = targets_labels

        print("GIfTI Dataset:")
        print(runallgii)

        if not args.surf:
            tsdata = runallnii # time-series data -> might want to use .copy(deep=T/F)
        else:
            tsdata = runallgii
        # eventually want to load only volume/surface:
        '''
        # VOLUME
        if not args.surf:
            print("Reading volumetric functional data")
            all_runs_bold_fname = os.path.join(args.output_dir, subj_name,
                                               subj_name + '_task-' + args.task + '_desc-preproc_bold.nii.gz')
        # SURFACE
        else:
            print("Reading surface-based functional data")
        '''
        ########################################################################
        ########################################################################

        ########################################################################
        ##################### Pre-Processing of Time-Series ####################
        ########################################################################

        # detrending is enabled
        if args.poly_detrend:
            poly_detrend(tsdata, polyord=args.poly_detrend, chunks_attr='chnks')
            # Event-related pre-processing is not event-related
            # some pre-processing is only meaningful when performed on the
            # full time-series and not on the segmented event samples. An
            # example is detrending that typically needs to be done on the
            # original, continuous time series

            # note: data normally stems from several runs and the assumption of a continuous linear trend
            # across all runs is not appropriate: poly_detrend(tsdata, polyord=args.poly_detrend)
        # no detrending
        else:
            pass

        # normalization is enabled
        if args.zscore:
            zscore(tsdata, chunks_attr='chnks') # This function behaves identical to ZScoreMapper. The only difference
                                                # is that z-scoring here is done in-place, potentially causing a
                                                # significant reduction of memory demands
        # no normalization
        else:
            pass
        ########################################################################
        ########################################################################

        ########################################################################
        ################################# GLM ##################################
        ########################################################################

        # simple average-sample approach is limited to block-designs with a clear
        # temporal separation of all signals of interest, whereas HRF modeling is
        # more suitable for experiments with fast stimulation alternation

        # add option to save betas for later use and bypass this step?

        print('####################')
        print('HRF MODELING STARTED')
        print('####################')
        if not args.lss:
            print('OLS is running...')
            evds = fit_event_hrf_model(tsdata,
                                       events,  # it is perfectly fine to have events that are not synchronized with
                                       # the TR. The labeling of events is taken from the 'events' list. Any
                                       # attribute(s) that are also in the dicts will be assigned as
                                       # condition labels in the output dataset
                                       # note that original_events and events are the same,
                                       time_attr='time_coords', # identifies at which timepoint each BOLD volume was acquired
                                       condition_attr=cond_attr  # name of the event attribute with condition
                                       # labels, which can be a list (e.g. ['targets', 'chunks'])

                                       # glmfit_kwargs=dict(model='ar1') # auto-regressive modeling
                                       # regr_attrs=['gsr']
                                       )
        else:
            print('LSS is running...')
            # note: '=' and 'copy()' make shallow copies of dictionaries. For deep copies use deepcopy

            for trial in range(0, len(events)):
                this_events = deepcopy(events[trial])
                events[trial]['targets'] = 'currenttrial' # changing only the current trial's label
                single_evds = fit_event_hrf_model(tsdata,
                                                  events,
                                                  time_attr='time_coords',
                                                  condition_attr=cond_attr
                                                  )
                # Keep only the current trial's beta:
                single_evds = single_evds[np.array([l in ['currenttrial']
                                                    for l in single_evds.targets], dtype='bool')]
                # at this point, single_evds is a single-sample Dataset
                single_evds.sa['targets'] = [this_events['targets']]
                if trial == 0:
                    evds = single_evds
                else:
                    evds = md.vstack((evds, single_evds))
                events[trial]['targets'] = this_events['targets'] # changing 'currenttrial' back to its original label
        print('####################')
        print('HRF MODELING ENDED')
        print('####################')

        ########################################################################
        ########################################################################

        ########################################################################
        ################### Ordering beta estimates for RSA ####################
        ########################################################################

        # order of trials gets mixed from this point -> should sort

        if args.rsa:
            # only for RSA, as there is no SVM classification with built-in z-scoring,
            # we'd want to z-score betas:
            if args.zscore:
                zscore(evds, chunks_attr='chunks')
            else:
                pass

            print('Ordering...')
            counter = 0
            for ord in stim_order:
                for srch in evds: # does this change evds too? deepcopy?
                    if srch.sa.targets == ord:
                        if counter == 0:
                            sorted_evds = srch
                        else:
                            sorted_evds = md.vstack((sorted_evds, srch))
                        # sorted_evds[counter] = srch
                        # sorted_evds[counter] = srch.copy(deep=True, sa=None, fa=None, a=None)
                        counter = counter + 1
                        break
            # evds = sorted_evds
        # before running classification (ROI-based/searchlight - both volume and surface), remove unwanted conditions:
        else:
            evds = evds[np.array([l in args.conditions_to_classify
                                  for l in evds.targets], dtype='bool')]

        ########################################################################
        ############################## ROI-based ############################## not on the surface yet
        ########################################################################
        if not args.searchlight:
            # the following was used when looping through multiple ROIs inside the "masks"
            # folder -> now analysis is on a single ROI
            '''
            ROIs = []
            ROIs = sorted(os.listdir(os.path.join(args.output_dir, 'masks')))
            for ROIloop in range(0, len(ROIs)):
                ROI_name = ROIs[ROIloop].split('.')[0]
            '''

            if args.rsa:
                print('ROI-based RSA is running...')
                mtgs = mean_group_sample(['targets'])
                mtds = sorted_evds.get_mapped(mtgs) # might never use

                mtcgs = mean_group_sample(['targets', 'chunks'])
                mtcds = sorted_evds.get_mapped(mtcgs)


                if args.dist == 'correlation':
                    dsm = rsa.PDist(square=True) #                                  correlation distance
                    RDM_title = 'Correlation'
                elif args.dist == 'euclidean':
                    dsm = rsa.PDist(square=True, pairwise_metric='euclidean') #     euclidean distance
                    RDM_title = 'Euclidean'
                elif args.dist == 'mahalanobis':
                    dsm = rsa.PDist(square=True, pairwise_metric='mahalanobis') #   mahalanobis distance
                    RDM_title = 'Mahalanobis'

                distances = dsm(sorted_evds) # <Dataset: #oftrialsx#oftrials@float64, <sa: chunks,onset,regressors,targets>>
                dist_mat = np.matrix(distances.samples)
                # dist_mat = 1-dist_mat  # changes distance to correlation r = 1-d

                plot_mtx(dist_mat, sorted_evds.sa.targets, 'ROI (' + args.mask + ') Pattern ' +
                         RDM_title + ' Distances') # sorted_evds.sa.targets is the same as distances.sa.targets
                draw2html()
            else:
                print('ROI-based Classification...')
                # evds = evds.get_mapped(mean_group_sample(['targets', 'chunks'])) # add functionality for averaging betas?

                ####################From Betas To Spatio-temporal Samples####################

                sens = cv_sensana(evds)
                sens_comb = sens.get_mapped(
                    maxofabs_sample())  # another way to combine sensitivity maps -> into a
                # single map. Note that sensitivities cannot be directly compared to each
                # other, even if they stem from the same algorithm and are just computed
                # on different datasets. In an analysis one would have to normalize them
                # first. PyMVPA offers, for example, l1_normed() and l2_normed() that can be used
                # be used in conjunction with FxMapper to do that as a post-processing step

                nimg = map2nifti(tsdata, sens_comb)
                nimg.to_filename(os.path.join(args.output_dir, subj_name,
                                              subj_name + '_task-' + args.task + '_' + args.mask +
                                              '_pattern-' + '-'.join(args.conditions_to_classify) + '.nii.gz'))

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
                html_str = html_str % (args.mask, cv_sensana.clf.ca.stats.as_string(description=True))
                subj_html.write(html_str)
        ########################################################################
        ############################# Searchlight ##############################
        ########################################################################
        else:

            #############################################################################
            ########## Making Spheric/Disc Searchlights for Classification/RSA ##########
            # VOLUME
            if not args.surf:
                print('Volumetric Searchlight Classification')
                sl = sphere_searchlight(cv, radius=args.searchlight, space='voxel_indices', nproc=args.nproc,
                                        postproc=mean_sample())
            # SURFACE
            else:
                print('Surface-Based Searchlight Classification')
                # SurfaceQueryEngine -> engine to use to discover the "neighborhood" of each feature
                # Loading the graymid (intermediate) surface (generated with mris_expand -thickness rh.white 0.5 graymid)
                fs_mid = surfread(os.path.join(args.bids_dir, 'derivatives', 'sourcedata',
                                                    'freesurfer', args.space, 'surf', args.hemi + 'h.graymid'))
                # radius = 5 (int) -> max of 5 neighboring vertices
                # radius = 5.0 (float) -> all vertices within a disc of radius 5 mm -> preferred
                qe = SurfaceQueryEngine(surface=fs_mid, radius=args.searchlight, distance_metric='dijkstra')
                print('Query Engine Successful')

                roi_ids = None # all nodes are used as searchlight centers

                # for RSA probably need to replace cv with dsm ('datameasure')
                sl = Searchlight(cv, queryengine=qe, postproc=mean_sample(), roi_ids=roi_ids, nproc=args.nproc)
            #############################################################################
            #############################################################################


            if args.rsa:
                print('Searchlight RSA is running...')
                # use sorted_evds rather than evds for RSA
            else:
                print("Searchlight Classification Started!")
                start_time = time.time()
                sl_res = sl(evds)
                print("--- %s seconds ---" % (time.time() - start_time))
                print("Searchlight Classification Ended!")

                # VOLUME
                if not args.surf:
                    print('Saving into NIfTI...')
                    # transforming error maps into accuracies: (no need to do this as CrossValidation's errorfx in modified
                    # sl_res.samples *= -1
                    # sl_res.samples += 1

                    niftiresults = map2nifti(tsdata, sl_res)
                    niftiresults.to_filename(os.path.join(args.output_dir, subj_name, subj_name + '_task-' + args.task +
                                                          '_searchlight-' + str(args.searchlight) + '_pattern-' +
                                                          '-'.join(args.conditions_to_classify) + '.nii.gz'))

                    # Add accuracy distribution plot to html?

                # SURFACE
                else:
                    print('Saving into GIfTI...')
                    if externals.exists('nibabel'):
                        print('externals exists: nibabel')
                        path_fn = os.path.join(args.output_dir, subj_name, subj_name + '_task-' + args.task +
                                               '_' + hemisphere +
                                               '_space-' + args.space +
                                               '_searchlight-' + str(int(args.searchlight)) + '_pattern-' +
                                               '-'.join(args.conditions_to_classify) + '.gii')
                        map2gifti(sl_res, path_fn)

        subj_html.close()

run_script('chmod -R 777 %s' % args.output_dir)
