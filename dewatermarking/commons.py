import os
import json
from datetime import datetime

import base64
import time
def generate_base64_timestamp_suffix():
    timestamp = int(time.time() )
    ts_int = int(timestamp)
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if ts_int == 0:
        return "0"
    result = []
    while ts_int > 0:
        ts_int, remainder = divmod(ts_int, 36)
        result.append(digits[remainder])
        
    # Reverse the list since we constructed it backwards
    return "".join(reversed(result))
class SaveMixin:
    def __init__(self, output_dir=None, name=None, action_type=None):
        if output_dir is None:
            raise ValueError("`output_dir` must not be None.")
        if name is None:
            raise ValueError("`name` must not be None.")
        if action_type is None:
            raise ValueError("`action_type` must not be None.")
        
        self.output_dir = output_dir
        self.name = name
        self.action_type=action_type
        self.display_name=name+"_"+action_type
        self._ensure_directory_exists(self.output_dir)
        self.filename = self.resolve_path()
        current_time = datetime.now()
        self.formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def resolve_path(self):
        if self.output_dir is None:
            raise ValueError("`output_dir` is not set.")
        if self.name is None:
            raise ValueError("`name` is not set.")
        os.makedirs(self.output_dir,exist_ok=True)
        return os.path.join(self.output_dir, f"{self.name}_{self.action_type}_{generate_base64_timestamp_suffix()}.json")
    
    def custom_parameters(self):
        return {}
    @property
    def parameters(self):
        params = {
            "time": self.formatted_time
        }
        custom_parameter = self.custom_parameters()
        return {**params,**custom_parameter}
    
    def save_outputs(self, outputs):
        with open(self.filename, "w") as f:
            json.dump({"parameters":self.parameters,
                "data":outputs}, f, indent=4)
