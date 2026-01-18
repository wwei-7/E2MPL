# import json
# import os
# import csv
# import make_args
# import time, datetime
# import shutil
# from nni.experiment import Experiment


# class Recorder:
#     def __init__(self):
#         pass

#     def step(self):
#         pass


# class ParameterRecorder(Recorder):
#     def __init__(self, tobe_adjust_parameters_dir):
#         super(ParameterRecorder, self).__init__()
#         # parameters in 'run_script.py'
#         self.tobe_adjust_parameters_dir = tobe_adjust_parameters_dir
#         self.list_len = {}
#         self.parameter_idx_mark = 0
#         self.adjust_finished_mark = {}
#         self.eof = False
#         for k in tobe_adjust_parameters_dir.keys():
#             self.list_len[k] = tobe_adjust_parameters_dir[k].__len__() - 1
#             self.adjust_finished_mark[k] = False
#         self.construct()

#     def construct(self):
#         # Get all arguments in the make_args.py
#         self.opt = make_args.args()
#         self.group = self.set_group()

#     def set_group(self) -> list:
#         """
#         Return the combinations of parameters for each experiment.
#         Returns:
#             root (2-d list): combinations of parameters
#         """
#         arg_num = len(self.tobe_adjust_parameters_dir)
#         # Hyper-parameters in the run_script.py: ['para1', 'para2', ...]
#         keys = list(self.tobe_adjust_parameters_dir.keys())
#         keys.reverse()
#         root = [[v] for v in self.tobe_adjust_parameters_dir[keys[0]]]
#         for i in range(1, len(keys)):
#             k = keys[i]
#             now_vertex = self.tobe_adjust_parameters_dir[k]
#             tmp_root = []
#             tmp_r = root.copy()
#             for arg in now_vertex:
#                 for r in tmp_r:
#                     t_r = r.copy()
#                     t_r.insert(0, arg)
#                     tmp_root.append(t_r)
#             root = tmp_root
#         return root

#     def step(self):
#         # Get now status
#         status = {k: self.group[self.parameter_idx_mark][i] for i, k in enumerate(self.tobe_adjust_parameters_dir.keys())}
       
#         # Update the status for the next experiment
#         idx = self.parameter_idx_mark
#         paras = self.group[idx]
#         for i, k in enumerate(self.list_len.keys()):
#                 self.opt.__dict__[k] = paras[i]

#         # Detect if last one experiment.
#         eof = self.parameter_idx_mark >= self.group.__len__() - 1
#         # print('*'*20, self.parameter_idx_mark, self.group.__len__(), eof, '*'*20)
        
#         if self.parameter_idx_mark < self.group.__len__() - 1:
#             self.parameter_idx_mark += 1
#         return self.opt, status, eof


# class ParameterAutoFinder(ParameterRecorder):
#     def __init__(self, tobe_adjust_parameters_dir):
#         super(ParameterAutoFinder, self).__init__(tobe_adjust_parameters_dir)
        
#     def construct(self):
#         # Get all arguments in the make_args.py
#         self.opt = make_args.args()
#         self.group = self.set_group()

#     def set_group(self) -> list:
#         """
#         Return the combinations of parameters for each experiment.
#         Returns:
#             root (2-d list): combinations of parameters
#         """
#         arg_num = len(self.tobe_adjust_parameters_dir)
#         # Hyper-parameters in the run_script.py: ['para1', 'para2', ...]
#         keys = list(self.tobe_adjust_parameters_dir.keys())
#         keys.reverse()
#         root = [[v] for v in self.tobe_adjust_parameters_dir[keys[0]]]
#         for i in range(1, len(keys)):
#             k = keys[i]
#             now_vertex = self.tobe_adjust_parameters_dir[k]
#             tmp_root = []
#             tmp_r = root.copy()
#             for arg in now_vertex:
#                 for r in tmp_r:
#                     t_r = r.copy()
#                     t_r.insert(0, arg)
#                     tmp_root.append(t_r)
#             root = tmp_root
#         return root

#     def step(self):
#         # Get now status
#         status = {k: self.group[self.parameter_idx_mark][i] for i, k in enumerate(self.tobe_adjust_parameters_dir.keys())}
       
#         # Update the status for the next experiment
#         idx = self.parameter_idx_mark
#         paras = self.group[idx]
#         for i, k in enumerate(self.list_len.keys()):
#                 self.opt.__dict__[k] = paras[i]

#         # Detect if last one experiment.
#         eof = self.parameter_idx_mark >= self.group.__len__() - 1
#         print('*'*20, self.parameter_idx_mark, self.group.__len__(), eof, '*'*20)
        
#         if self.parameter_idx_mark < self.group.__len__() - 1:
#             self.parameter_idx_mark += 1
#         return self.opt, status, eof



# class AccFileWriter:
#     def __init__(self, dir, parameters_dir):
#         self.dir = dir
#         if not os.path.exists(dir):
#             os.mkdir(dir)
#         # self.date = time.strftime("%Y-%m-%d-%h-", time.localtime())
#         self.date = str(datetime.datetime.now())
#         # self.transfer = '%s2%s' %(parameters_dir['source_domain'], parameters_dir['target_domain'])
#         self.record_path = os.path.join(dir, self.date + '-'+parameters_dir['metric'][0]+'-'+parameters_dir['data_name'][0], )
#         self.record_path = self.record_path.replace(' ', '-')
#         if not os.path.exists(self.record_path):
#             os.mkdir(self.record_path)
#         time_now = time.asctime(time.localtime(time.time()))
#         # self.clear_empty_history()
#         self.file_path = os.path.join(self.record_path, 'results-(%s).csv'%str(time_now))
#         self.keys = list(parameters_dir.keys())
#         self.keys.append('acc')
#         self.keys.append('val_acc')
#         self.keys = ['id'] + self.keys
#         with open(self.file_path, 'w', encoding='utf-8') as res_file:
#             writer = csv.writer(res_file)
#             writer.writerow(self.keys)

#     def clear_empty_history(self):
#         print('clearing the history in the \'%s\'...' % self.file_path)
#         for f in os.listdir(self.file_path):
#             if '.csv' in f:
#                 csv_f = os.path.join(self.file_path, f)
#                 with open(csv_f, 'r', encoding='utf-8') as file:
#                     reader = csv.reader(file)
#                     if reader.__len__() <= 1:
#                         os.remove(csv_f)

#     def update(self, results, now_parmeters_dir, ID):
#         acc_, h_ = results['aver_accuracy'], results['aver_h']
#         acc = '%.3f+-%.2f' % (acc_, h_)
#         para = [now_parmeters_dir[k] for k in self.keys if k not in ['id', 'acc', 'val_acc']]
#         para = [ID] + para
#         with open(self.file_path, 'a+', encoding='utf-8') as res_file:
#             writer = csv.writer(res_file)
#             para.append(acc)
#             para.append('%.3f'%results['val_acc'])
#             writer.writerow(para)

# from executor import callback_executor, callback_executor_tune
# class ExperimentServer:
#     def __init__(self, parameters_dir, func_tobe_excuted, auto_search=False, paras_space={}, results_dir=''):
#         self.now_id = 0
#         self.adjust_parameters_dir = parameters_dir
#         self.func_tobe_excuted = func_tobe_excuted
#         assert os.path.exists(results_dir) and os.path.isdir(results_dir)
#         self.results_dir = os.getcwd() if results_dir is None or not os.path.exists(results_dir) else results_dir
#         self.file_writer = AccFileWriter(dir='{}/adjust_parameters'.format(self.results_dir), parameters_dir=parameters_dir)
#         self.record_path = self.file_writer.record_path
#         self.parameter_record = ParameterRecorder(parameters_dir)
#         self.auto_search = auto_search
#         self.paras_space = paras_space
#         self.construct()
#         self.now_args_json = ''

#     def construct(self):
#         self.opt = self.parameter_record.opt
#         self.opt.record_path = self.record_path

#     def run_nni_experiment(self):
#         exp = Experiment('local')
#         # ------------------------------- Do Not Change ----------------------------------
#         # Set id of the experiment, to prevent the duplicate experiment in nni framework.
#         exp.id = ''.join([str(ord(c)) for c in [k for k in str(datetime.datetime.now()) if k in '0123456789']])
#         exp.config.experiment_name = "{},{}-way,{}-shot,{}-to-{}".format(self.opt.metric, 
#                 self.opt.way_num, self.opt.shot_num, 
#                 self.opt.source_domain, self.opt.target_domain)
#         exp.config.search_space = self.paras_space
#         exp.config.trial_command = 'python executor.py --mode=tune --argspath={}'.format(str(self.now_args_json))
#         exp.config.trial_code_directory = '.'
#         # ------------------------------- Do not Change ----------------------------------
#         # ------------------------------- Can Be Modified to Change nni Configuration ----------------------------------
#         exp.config.trial_gpu_number = 1
#         # Set how many experiment concurrently to be run
#         exp.config.trial_concurrency = 2
        
        
#         # Set early stop method to save time, see: https://nni.readthedocs.io/zh/stable/hpo/assessors.html
#         exp.config.assessor.name = 'Medianstop'
#         exp.config.max_trial_number = 25
#         exp.config.max_experiment_duration = '24h'
#         exp.config.tuner.name = 'Anneal'
#         exp.config.tuner.class_args = {
#             'optimize_mode': 'maximize',
#         }
#         exp.config.training_service.platform = 'local'
#         exp.config.training_service.use_active_gpu = True
#         exp.config.training_service.gpu_indices = os.environ['CUDA_VISIBLE_DEVICES']
        
#         # Decide how many experiment run on the one GPU
#         exp.config.training_service.max_trial_number_per_gpu = 1
#         # Set network port to show visualization results, see: https://nni.readthedocs.io/zh/stable/experiment/web_portal/web_portal.html
#         return exp.run(8200)
#         # ------------------------------- Can Be Modified to Change nni Configuration ----------------------------------
#     def run(self):
#         eof = False
#         while not eof:
#             try:
#                 print('training...')
#                 self.opt, now_parmeters_dir, eof = self.parameter_record.step()
#                 # make experiment dir
#                 self.opt.exp_id = self.now_id
#                 self.opt.auto_tune = bool(self.auto_search)
#                 self.opt.outf = self.file_writer.record_path
#                 self.now_args_json = os.path.join(self.file_writer.record_path, 'all_args-{}.json'.format(self.now_id))
#                 json.dump(vars(self.opt), open(self.now_args_json, 'w'))
#                 # Change it and put into the middle-ware 'callback_executor' in 'executor.py'
#                 if not self.auto_search:
#                     callback_executor(self.opt, self.func_tobe_excuted)
#                 else:
#                     self.run_nni_experiment()
#                 self.now_id += 1
#             except BaseException as E:
#                 import traceback
#                 with open(self.file_writer.file_path, 'r', encoding='utf-8') as csv_file:
#                     readline = csv.reader(csv_file)
#                     for i, line in enumerate(readline):
#                         if i >= 1:
#                             exit()
#                     os.remove(self.file_writer.file_path)
#                 for f in os.listdir(self.record_path):
#                     if os.path.isdir(os.path.join(self.record_path, f)):
#                         for k in os.listdir(os.path.join(self.record_path, f)):
#                             try:
#                                 os.close(open(os.path.join(self.record_path, f, k), "r"))
#                                 os.remove(os.path.join(self.record_path, f, k))
#                             except:
#                                 print(k)
#                         shutil.rmtree(os.path.join(self.record_path, f), ignore_errors=True)
#                     # os.removedirs(os.path.join(self.record_path, f))
#                 print(self.record_path)
#                 shutil.rmtree(self.record_path, ignore_errors=True)
#                 print(traceback.print_exc())
#                 exit()
import json
import os
import csv
import make_args
import time, datetime
import shutil
from nni.experiment import Experiment


class Recorder:
    def __init__(self):
        pass

    def step(self):
        pass


class ParameterRecorder(Recorder):
    def __init__(self, tobe_adjust_parameters_dir):
        super(ParameterRecorder, self).__init__()
        # parameters in 'run_script.py'
        self.tobe_adjust_parameters_dir = tobe_adjust_parameters_dir
        self.list_len = {}
        self.parameter_idx_mark = 0
        self.adjust_finished_mark = {}
        self.eof = False
        for k in tobe_adjust_parameters_dir.keys():
            self.list_len[k] = tobe_adjust_parameters_dir[k].__len__() - 1
            self.adjust_finished_mark[k] = False
        self.construct()

    def construct(self):
        # Get all arguments in the make_args.py
        self.opt = make_args.args()
        self.group = self.set_group()

    def set_group(self) -> list:
        """
        Return the combinations of parameters for each experiment.
        Returns:
            root (2-d list): combinations of parameters
        """
        arg_num = len(self.tobe_adjust_parameters_dir)
        # Hyper-parameters in the run_script.py: ['para1', 'para2', ...]
        keys = list(self.tobe_adjust_parameters_dir.keys())
        keys.reverse()
        root = [[v] for v in self.tobe_adjust_parameters_dir[keys[0]]]
        for i in range(1, len(keys)):
            k = keys[i]
            now_vertex = self.tobe_adjust_parameters_dir[k]
            tmp_root = []
            tmp_r = root.copy()
            for arg in now_vertex:
                for r in tmp_r:
                    t_r = r.copy()
                    t_r.insert(0, arg)
                    tmp_root.append(t_r)
            root = tmp_root
        return root

    def step(self):
        # Get now status
        status = {k: self.group[self.parameter_idx_mark][i] for i, k in enumerate(self.tobe_adjust_parameters_dir.keys())}
       
        # Update the status for the next experiment
        idx = self.parameter_idx_mark
        paras = self.group[idx]
        for i, k in enumerate(self.list_len.keys()):
                self.opt.__dict__[k] = paras[i]

        # Detect if last one experiment.
        eof = self.parameter_idx_mark >= self.group.__len__() - 1
        # print('*'*20, self.parameter_idx_mark, self.group.__len__(), eof, '*'*20)
        
        if self.parameter_idx_mark < self.group.__len__() - 1:
            self.parameter_idx_mark += 1
        return self.opt, status, eof


class ParameterAutoFinder(ParameterRecorder):
    def __init__(self, tobe_adjust_parameters_dir):
        super(ParameterAutoFinder, self).__init__(tobe_adjust_parameters_dir)
        
    def construct(self):
        # Get all arguments in the make_args.py
        self.opt = make_args.args()
        self.group = self.set_group()

    def set_group(self) -> list:
        """
        Return the combinations of parameters for each experiment.
        Returns:
            root (2-d list): combinations of parameters
        """
        arg_num = len(self.tobe_adjust_parameters_dir)
        # Hyper-parameters in the run_script.py: ['para1', 'para2', ...]
        keys = list(self.tobe_adjust_parameters_dir.keys())
        keys.reverse()
        root = [[v] for v in self.tobe_adjust_parameters_dir[keys[0]]]
        for i in range(1, len(keys)):
            k = keys[i]
            now_vertex = self.tobe_adjust_parameters_dir[k]
            tmp_root = []
            tmp_r = root.copy()
            for arg in now_vertex:
                for r in tmp_r:
                    t_r = r.copy()
                    t_r.insert(0, arg)
                    tmp_root.append(t_r)
            root = tmp_root
        return root

    def step(self):
        # Get now status
        status = {k: self.group[self.parameter_idx_mark][i] for i, k in enumerate(self.tobe_adjust_parameters_dir.keys())}
       
        # Update the status for the next experiment
        idx = self.parameter_idx_mark
        paras = self.group[idx]
        for i, k in enumerate(self.list_len.keys()):
                self.opt.__dict__[k] = paras[i]

        # Detect if last one experiment.
        eof = self.parameter_idx_mark >= self.group.__len__() - 1
        print('*'*20, self.parameter_idx_mark, self.group.__len__(), eof, '*'*20)
        
        if self.parameter_idx_mark < self.group.__len__() - 1:
            self.parameter_idx_mark += 1
        return self.opt, status, eof



class AccFileWriter:
    def __init__(self, dir, parameters_dir):
        self.dir = dir
        if not os.path.exists(dir):
            os.mkdir(dir)
        # self.date = time.strftime("%Y-%m-%d-%h-", time.localtime())
        self.date = str(datetime.datetime.now())
        # self.transfer = '%s2%s' %(parameters_dir['source_domain'], parameters_dir['target_domain'])
        self.record_path = os.path.join(dir, self.date + '-'+parameters_dir['metric'][0]+'-'+parameters_dir['data_name'][0], )
        self.record_path = self.record_path.replace(' ', '-')
        if not os.path.exists(self.record_path):
            os.mkdir(self.record_path)
        time_now = time.asctime(time.localtime(time.time()))
        # self.clear_empty_history()
        self.file_path = os.path.join(self.record_path, 'results-(%s).csv'%str(time_now))
        self.keys = list(parameters_dir.keys())
        self.keys.append('acc')
        self.keys.append('val_acc')
        self.keys = ['id'] + self.keys
        with open(self.file_path, 'w', encoding='utf-8') as res_file:
            writer = csv.writer(res_file)
            writer.writerow(self.keys)

    def clear_empty_history(self):
        print('clearing the history in the \'%s\'...' % self.file_path)
        for f in os.listdir(self.file_path):
            if '.csv' in f:
                csv_f = os.path.join(self.file_path, f)
                with open(csv_f, 'r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    if reader.__len__() <= 1:
                        os.remove(csv_f)

    def update(self, results, now_parmeters_dir, ID):
        acc_, h_ = results['aver_accuracy'], results['aver_h']
        acc = '%.3f+-%.2f' % (acc_, h_)
        para = [now_parmeters_dir[k] for k in self.keys if k not in ['id', 'acc', 'val_acc']]
        para = [ID] + para
        with open(self.file_path, 'a+', encoding='utf-8') as res_file:
            writer = csv.writer(res_file)
            para.append(acc)
            para.append('%.3f'%results['val_acc'])
            writer.writerow(para)

from executor import callback_executor, callback_executor_tune
class ExperimentServer:
    def __init__(self, parameters_dir, func_tobe_excuted, auto_search=False, paras_space={}, results_dir=''):
        self.now_id = 0
        self.adjust_parameters_dir = parameters_dir
        self.func_tobe_excuted = func_tobe_excuted
        assert os.path.exists(results_dir) and os.path.isdir(results_dir)
        self.results_dir = os.getcwd() if results_dir is None or not os.path.exists(results_dir) else results_dir
        self.file_writer = AccFileWriter(dir='{}/adjust_parameters'.format(self.results_dir), parameters_dir=parameters_dir)
        self.record_path = self.file_writer.record_path
        self.parameter_record = ParameterRecorder(parameters_dir)
        self.auto_search = auto_search
        self.paras_space = paras_space
        self.construct()
        self.now_args_json = ''

    def construct(self):
        self.opt = self.parameter_record.opt
        self.opt.record_path = self.record_path

    def run_nni_experiment(self):
        exp = Experiment('local')
        # ------------------------------- Do Not Change ----------------------------------
        # Set id of the experiment, to prevent the duplicate experiment in nni framework.
        exp.id = ''.join([str(ord(c)) for c in [k for k in str(datetime.datetime.now()) if k in '0123456789']])
        exp.config.experiment_name = "{},{}-way,{}-shot,{}-to-{}".format(self.opt.metric, 
                self.opt.way_num, self.opt.shot_num, 
                self.opt.source_domain, self.opt.target_domain)
        exp.config.search_space = self.paras_space
        exp.config.trial_command = 'python executor.py --mode=tune --argspath={}'.format(str(self.now_args_json))
        exp.config.trial_code_directory = '.'
        # ------------------------------- Do not Change ----------------------------------
        # ------------------------------- Can Be Modified to Change nni Configuration ----------------------------------
        exp.config.trial_gpu_number = 1
        # Set how many experiment concurrently to be run
        exp.config.trial_concurrency = 8
        
        
        # Set early stop method to save time, see: https://nni.readthedocs.io/zh/stable/hpo/assessors.html
        exp.config.assessor.name = 'Medianstop'
        exp.config.max_trial_number = 20
        exp.config.max_experiment_duration = '24h'
        exp.config.tuner.name = 'Anneal'
        exp.config.tuner.class_args = {
            'optimize_mode': 'maximize',
        }
        exp.config.training_service.platform = 'local'
        exp.config.training_service.use_active_gpu = True
        exp.config.training_service.gpu_indices = os.environ['CUDA_VISIBLE_DEVICES']
        
        # Decide how many experiment run on the one GPU
        exp.config.training_service.max_trial_number_per_gpu = 2
        # Set network port to show visualization results, see: https://nni.readthedocs.io/zh/stable/experiment/web_portal/web_portal.html
        return exp.run(8205)
        # ------------------------------- Can Be Modified to Change nni Configuration ----------------------------------
    def run(self):
        eof = False
        while not eof:
            try:
                print('training...')
                self.opt, now_parmeters_dir, eof = self.parameter_record.step()
                # make experiment dir
                self.opt.exp_id = self.now_id
                self.opt.auto_tune = bool(self.auto_search)
                self.opt.outf = self.file_writer.record_path
                self.now_args_json = os.path.join(self.file_writer.record_path, 'all_args-{}.json'.format(self.now_id))
                json.dump(vars(self.opt), open(self.now_args_json, 'w'))
                # Change it and put into the middle-ware 'callback_executor' in 'executor.py'
                if not self.auto_search:
                    callback_executor(self.opt, self.func_tobe_excuted)
                else:
                    self.run_nni_experiment()
                self.now_id += 1
            except BaseException as E:
                import traceback
                with open(self.file_writer.file_path, 'r', encoding='utf-8') as csv_file:
                    readline = csv.reader(csv_file)
                    for i, line in enumerate(readline):
                        if i >= 1:
                            exit()
                    os.remove(self.file_writer.file_path)
                for f in os.listdir(self.record_path):
                    if os.path.isdir(os.path.join(self.record_path, f)):
                        for k in os.listdir(os.path.join(self.record_path, f)):
                            try:
                                os.close(open(os.path.join(self.record_path, f, k), "r"))
                                os.remove(os.path.join(self.record_path, f, k))
                            except:
                                print(k)
                        shutil.rmtree(os.path.join(self.record_path, f), ignore_errors=True)
                    # os.removedirs(os.path.join(self.record_path, f))
                print(self.record_path)
                shutil.rmtree(self.record_path, ignore_errors=True)
                print(traceback.print_exc())
                exit()
