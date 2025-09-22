import datetime
import errno
import numpy as np
import pandas as pd
import gc
from collections import deque
import os

class Logger:
    def __init__(self, curdir: str, params: dict, log_params: dict, train: bool, buffer_size: int):
        OUTPUT_FILE_EXTENSION = ".csv"

        mode = "train" if train else "eval"
        time_now = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

        base_dir = os.path.join(curdir, "runs/" + mode)
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)
        output_dir = os.path.join(base_dir, mode + "_" + time_now)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        filename = log_params[mode + "_output_file"].replace("-", "_") + '_' + time_now + '_other' + OUTPUT_FILE_EXTENSION
        self.output_file = os.path.join(output_dir, filename)
        
        self.metrics = tuple(self._get_metrics(params))
        self.table = pd.DataFrame(columns=self.metrics)
        self.buffer_size = buffer_size
    
    def _get_metrics(self, params):
        metrics = ["Episode", "Tick"]

        for a in range(params["learners"]):
            metrics.append(f"agent-{a}_action")
        for a in range(params["learners"]):
            metrics.append(f"agent-{a}_nest")
        for a in range(params["learners"]):
            metrics.append(f"agent-{a}_food")

        return metrics
    
    def _write_to_csv(self):
        if os.path.isfile(self.output_file): # check se il file esiste
            with open(self.output_file, 'a') as f:
                self.table.to_csv(f, header=False, sep=',', index=False)
        else: # check se il file non esiste
            self.table.to_csv(self.output_file, sep=',', index=False)
        return True

    def load_values(self, values):
        assert(isinstance(values, list) or isinstance(values, tuple)), "Error: values must be of type list or tuple!"
        values = deque(values)
        while(len(values) != 0):
            # Check if full
            quantity = self.buffer_size - self.table.shape[0]
            if quantity == 0:
                flag = self._write_to_csv()
                if flag:
                    self._reinit()
            else:
                if quantity >= len(values):
                    tmp = [values.popleft() for _ in range(len(values))]
                else:
                    tmp = [values.popleft() for _ in range(quantity)]
                self._add_rows(tmp)

    def load_value(self, value):
        assert(isinstance(value, list) or isinstance(value, tuple)), "Error: value must be of type list or tuple!"
        quantity = self.buffer_size - self.table.shape[0]
        if quantity == 0:
            flag = self._write_to_csv()
            if flag:
                self._reinit()
        self._add_rows([value])

    def _add_rows(self, vals):
        tmp = pd.DataFrame(vals, columns=self.metrics)
        if self.table.shape[0] == 0:
            self.table = self.table.combine_first(tmp)
        else:
            self.table = pd.concat([self.table, tmp], ignore_index=True)
    
    def _delete_table(self):
        self.table = None
        gc.collect()

    def _reinit(self):
        self._delete_table()
        self.table = pd.DataFrame(columns=self.metrics)
    
    def empty_table(self):
        if self.table.shape[0] > 0:
            self._write_to_csv()
        self._delete_table()