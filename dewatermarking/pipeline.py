from abc import ABC, abstractmethod
from typing import List, Dict, Any
import dewatermarking.generate as generate
import dewatermarking.detect as detect
import dewatermarking.rephrase as rephrase
from dewatermarking.evaluate import EmbeddingSimilarityStrategy,WatermarkDetectionStrategy
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from accelerate import Accelerator
accelerator = Accelerator()
from dataclasses import dataclass
from typing import Optional
import json
import os
import argparse
class PersistCheckpoint:
    def __init__(self,path,checkpoint):
        self.path = path
        self.checkpoint = checkpoint
    def persist(self):
        with open(self.path, "w") as f:
            json.dump(self.checkpoint, f,indent=4)
    def history(self,key,value):
        self.checkpoint["history"][key] = value
        self.persist()
    def get_history(self,key):
        return self.checkpoint["history"][key]
    def is_in_history(self,key):
        return key in self.checkpoint["history"]
    def set_stats(self,key,value):
        if "stats" not in self.checkpoint:
            self.checkpoint["stats"] = {}
        self.checkpoint["stats"][key] = value
        self.persist()
def detect_step(path,output_dir,type,name,persist_checkpoint):
    if not persist_checkpoint.is_in_history(name):
        print(f"Detecting {name} {type}")
        detect_args = argparse.Namespace(**{
            "path": path,
            "type": type,
            "output_dir": f"{output_dir}/detect",
            "name": name
        })
        result_dectect_file = detect.run(detect_args)
        persist_checkpoint.history(name,result_dectect_file)
    else:
        print(f"Skipping {name} because it is already in history")
    result_dectect_file = persist_checkpoint.get_history(name)
    if not persist_checkpoint.is_in_history("p value "+name):
        detect_watermarking_strategy = WatermarkDetectionStrategy()
        stats = detect_watermarking_strategy.evaluate(result_dectect_file)
        persist_checkpoint.set_stats("p value "+name,stats)
        persist_checkpoint.history("p value "+name,stats)
def rephrased_step(path,output_dir,name,method,persist_checkpoint):
    if not persist_checkpoint.is_in_history(name):
        print(f"Rephrasing {name} {method}")
        rephrased_no_watermarking_args = argparse.Namespace(**{
            "dataset_file": path,
            "column_name": "completion",
        "batch_size": 16,
        "max_new_tokens": 300,
        "output_dir": f"{output_dir}/rephrase",
        "name": name,
        "method": method,
        "model_id": "meta-llama/Meta-Llama-3-8B"
         })
        rephrased_data_file = rephrase.run(rephrased_no_watermarking_args)  
        persist_checkpoint.history(name,rephrased_data_file)
    else:
        print(f"Skipping {name} because it is already in history")
    rephrased_data_file = persist_checkpoint.get_history(name)
    if method != "no_watermarking":
        detect_step(rephrased_data_file,output_dir,"current","detect_current_"+name,persist_checkpoint)
    detect_step(rephrased_data_file,output_dir,"previous","detect_previous_"+name,persist_checkpoint)
    embedding_similarity_step(rephrased_data_file,"similarity_"+name,persist_checkpoint)
def embedding_similarity_step(path,name,persist_checkpoint):
    if persist_checkpoint.is_in_history(name):
        return
    embedding_similarity_strategy = EmbeddingSimilarityStrategy()
    stats = embedding_similarity_strategy.evaluate(path)
    persist_checkpoint.set_stats("similarity "+name,stats)
    persist_checkpoint.history("similarity "+name,stats)

def generate_step(dataset_file,output_dir,name,method,persist_checkpoint,limit=100):
    if not persist_checkpoint.is_in_history(name):
        print(f"Generating {name} {method}")
        generate_args = argparse.Namespace(**{
            "model_id": "meta-llama/Meta-Llama-3-8B",
            "dataset_file": dataset_file,
            "column_name": "gold_completion",
            "batch_size": 16,
            "max_new_tokens": 300,
            "output_dir": f"{output_dir}/generate",
            "name": name,
            "method": method,
            "limit": limit
        })
        result_dataset_file = generate.run(generate_args)
        persist_checkpoint.history(name,result_dataset_file)
    else:
        print(f"Skipping {name} because it is already in history")
    result_dataset_file = persist_checkpoint.get_history(name)
    detect_step(result_dataset_file,output_dir,"current","detect_"+name,persist_checkpoint)
    rephrased_step(result_dataset_file,output_dir,"rephrase_gumbel_"+name,"gumbel",persist_checkpoint)
    rephrased_step(result_dataset_file,output_dir,"rephrase_red_green_"+name,"red_green",persist_checkpoint)
    rephrased_step(result_dataset_file,output_dir,"rephrase_no_watermarking_"+name,"no_watermarking",persist_checkpoint)
def run(dataset_files,output_dir,start_from_checkpoint=None,limit=100):
    if start_from_checkpoint is not None and os.path.exists(start_from_checkpoint):
        checkpoint = json.load(open(start_from_checkpoint))
    else:
        checkpoint = {}
    history = checkpoint["history"] if "history" in checkpoint else {}
    checkpoint["history"] = history
    persist_checkpoint = PersistCheckpoint(start_from_checkpoint,checkpoint)
    for name,dataset_file in dataset_files:
        torch.cuda.empty_cache()
        generate_step(dataset_file,output_dir,f"{name}_gumbel","gumbel",persist_checkpoint,limit=limit)
        torch.cuda.empty_cache()
        generate_step(dataset_file,output_dir,f"{name}_red_green","red_green",persist_checkpoint,limit=limit)


