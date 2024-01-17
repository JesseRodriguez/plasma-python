'''
This file contains a class that will help you figure out which permutation of
the available signals you would like to use.
'''

from __future__ import print_function
import plasma.global_vars as g
from os import listdir  # , remove
import time
import sys
import os
import shutil
from itertools import combinations
from functools import partial

import numpy as np
import pathos.multiprocessing as mp

from plasma.utils.processing import append_to_filename
from plasma.utils.diagnostics import print_shot_list_sizes
from plasma.primitives.shots import ShotList
from plasma.utils.downloading import mkdirdepth
from plasma.utils.hashing import myhash_signals

class DatasetBuilder(object):
    def __init__(self, conf):
        self.conf = conf
        self.record_dir = os.path.join(conf['paths']['output_path'],'dataset_info/')


    def Check_Signal_Group_Permute(self, signal_list_permute, signal_list_fixed,\
            max_tol = False, record_name = 'Possible_Datasets.txt',\
            verbose = False, cpu_use = 0.8):
        """
        Runs Check_Signal_Group on every permutation of signal_list_permute,
        with signal_list_fixed being filled with elements that must be included
        in the datasets.

        Args:
            signal_list_permute: list of str keys to all_signals dict, want
                                 these to be permuted
            signal_list_fixed: list of str keys that MUST be included
            max_tol: bool, true if the user wants to use the maximum tolerance
                     all_signals dict
        """
        n = len(signal_list_permute)
        for r in range(n):
            for subset in combinations(signal_list_permute, r+1):
                self.Check_Signal_Group(signal_list_fixed+list(subset),\
                        max_tol, record_name, verbose, cpu_use)


    def Check_Signal_Group(self, signal_list, max_tol = False, record_name =\
            'Possible_Datasets.txt', verbose = False, cpu_use = 0.8):
        """
        Takes a list of keys to the all_signals dict and checks
        that list to see the size of the possible dataset

        Args:
            signal_list: list of str keys to all_signals
            max_tol: bool, true if the user wants to use the maximum tolerance
                     all_signals dict
        """
        shot_files = self.conf['paths']['shot_files']
        signals = {}
        all_signals = self.conf['paths']['all_signals_dict']
        for signal in signal_list:
            signals[signal] = all_signals[signal]
        shot_list = ShotList()
        shot_list.load_from_shot_list_files_objects(shot_files, signals.values())
        used_shots = ShotList()
        print('Checking signal group:')
        print(signal_list)
        print(80*'-')

        h = myhash_signals(signals.values())
        if os.path.exists(os.path.join(self.record_dir, record_name)):
            with open(os.path.join(self.record_dir, record_name), 'r') as file:
                report = file.read()
            if str(h) in report:
                print("This group of signals has already been checked!")
                return

        check_single_shot_verb = partial(self.check_single_shot,\
                                    verbose=verbose)
        assert cpu_use < 1
        use_cores = max(1, int((cpu_use)*mp.cpu_count()))
        pool = mp.Pool(use_cores)
        print('Running in parallel on {} processes'.format(pool._processes))
        start_time = time.time()
        for (i, shot) in enumerate(pool.imap_unordered(\
                check_single_shot_verb, shot_list)):
            sys.stdout.write('\r{}/{}'.format(i, len(shot_list)))
            used_shots.append_if_valid(shot)

        pool.close()
        pool.join()
        print('\nFinished checking {} shots in {} seconds'.format(
            len(shot_list), time.time() - start_time))
        print('This signal set yields {} shots ({} disruptive shots)'.format(
            len(used_shots), used_shots.num_disruptive()))
        print('Omitted {} shots of {} total shots'.format(
            len(shot_list) - len(used_shots), len(shot_list)))
        print('Omitted {} disruptive shots of {} total disruptive shots'.format(
                shot_list.num_disruptive() - used_shots.num_disruptive(),
                shot_list.num_disruptive()))

        if len(used_shots) == 0:
            print("WARNING: All shots were omitted, please ensure raw data "
                  " is complete and available at {}.".format(
                      self.conf['paths']['signal_prepath']))

        print(20*'\n')

        with open(os.path.join(self.record_dir, record_name), 'a') as file:
            file.write(80*'-'+'\n')
            file.write("For the following set of signals:\n")
            file.write(str(signal_list)+'\n\n')
            file.write('The valid dataset contains '+\
                    '{} shots ({} disruptive shots)\n'.\
                    format(len(used_shots), used_shots.num_disruptive()))
            file.write('Signal group hash:'+str(h)+'\n\n\n')

        self.sort_dataset_file(os.path.join(self.record_dir, record_name))

        return


    def check_single_shot(self, shot, verbose = False):
        """
        Loads signal data for a single shot to find out if a set of signals
        yields a valid shot.
        """
        shot.preprocess(self.conf, verbose, just_checking = True)
        shot.make_light()

        return shot
        

    def sort_dataset_file(self, file_path):
        """
        Sorts the dataset file by number of shots

        Args:
            file_path: str, path to dataset info file
        """
        num_files_preamble = 'The valid dataset contains '

        with open(file_path, 'r') as file:
            data = file.read()

        dataset_info = []
        num_shots = []
        data_len = len(data)
        L_s = len(num_files_preamble)
        for i in range(data_len-L_s+1):
            if data[i:i+L_s] == num_files_preamble:
                j = 0
                found_start = False
                while not found_start:
                    if data[i+L_s-j] == '-':
                        start = i+L_s-j
                        found_start = True
                    j += 1

                j = 0
                found_end = False
                while not found_end:
                    if data[i+L_s+j] == '-':
                        end = i+L_s+j
                        found_end = True
                    j += 1
                    if i+L_s+j == data_len - 1:
                        end = data_len - 1
                        found_end = True
                dataset_info.append(data[start-79:end])

                j = 0
                found_end = False
                while not found_end:
                    if data[i+L_s+j] == ' ':
                        end = i+L_s+j
                        found_end = True
                    j += 1
                num_shots.append(int(data[i+L_s:end]))
                
        combined = sorted(zip(num_shots, dataset_info), reverse = True)
        num_shots_sorted, dataset_info_sorted = zip(*combined)
        num_shots_sorted = list(num_shots_sorted)
        dataset_info_sorted = list(dataset_info_sorted)
        
        backup_path = file_path+'.backup'
        shutil.copy(file_path, backup_path)
        modified_file_path = file_path + '.modified'
        try:
            with open(modified_file_path, 'w') as modified_file:
                for info in dataset_info_sorted:
                    modified_file.write(info)
        except Exception as e:
            print("Exception:")
            print(e)
            print("Check backup dataset info file to make sure it's ok.")

        os.remove(file_path)
        os.rename(modified_file_path, file_path)
        os.remove(backup_path)

        return
