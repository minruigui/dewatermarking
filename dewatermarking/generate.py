import os
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
from dewatermarking.wm import GumbelSoftGeneratorNg
from dewatermarking.watermark.extended_watermark_processor import WatermarkLogitsProcessor,WatermarkDetector
from dewatermarking.mersenne import mersenne_rng
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dewatermarking.commons import SaveMixin
# Set environment variables for caching
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
class TextDataset(Dataset):
    def __init__(self, file_path, column_name, fn=None,limit=200):
        with open(file_path, 'r') as f:
            self.data = [json.loads(x) for x in f.read().strip().split("\n")]
        self.column_name = column_name
        self.fn = fn
        if self.fn is None:
            self.fn = lambda x:x
        if limit:
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        if not self.column_name and "prefix" in row:
            return self.fn(row["prefix"]), ""
        elif "prefix" in row and self.column_name in row:
            return self.fn(row["prefix"]), row[self.column_name]
        else:
            return self.fn(row["prefix"]), ""
class Strategy(SaveMixin):
    def __init__(self, model, tokenizer, max_new_tokens,dataset_file, column_name,device,accelerator,output_dir="./",name="",verbose=False):
        super().__init__(output_dir=output_dir,name=name,action_type="generate")
        model.resize_token_embeddings(len(tokenizer))
        self.accelerator=accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_file = dataset_file
        self.max_new_tokens=max_new_tokens
        self.column_name = column_name
        self.device= device
        self.verbose=verbose
        print(f"output file path:{self.filename}")
        self.dataset = TextDataset(self.dataset_file, self.column_name,lambda x:x)


    def load_dataset(self):
        with open(self.dataset_file, 'r') as f:
            return  [json.loads(x) for x in f.read().strip().split("\n")]

    def custom_parameters(self):
        return {"dataset_size":len(self.dataset)}
    def process(self, text):
        raise NotImplementedError("Subclasses should implement this method.")
    def convert_data_to_format(self,dd):
        return dd

    def execute(self,batch_size=8,save_interval=10):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        outputs = []
        processed_count = 0
        dataloader = self.accelerator.prepare(dataloader)
        
        is_main_process = self.accelerator.is_local_main_process
        if is_main_process:
            pbar = tqdm(total=len(dataloader), desc="Processing")
        for batch in dataloader:
            prefixes, gold_completions = batch
            results = self.process(prefixes)
            for prefix, result, gold_completion in zip(prefixes, results, gold_completions):
                output_dict = {
                    "prefix": prefix,
                    "completion": result,
                }
                if gold_completion != "":
                    output_dict["gold_completion"] = gold_completion
                    
                outputs.append(output_dict)
                if self.verbose:
                    print(output_dict)
            processed_count += 1
            if is_main_process:
                pbar.update(1)
                
            if processed_count % save_interval == 0:
                self.save_outputs(outputs)
        self.save_outputs(outputs)

class FixRandomGumbelStrategy(Strategy):
    name = "fix_random_gumbel"
    def __init__(self, model, tokenizer,max_new_tokens, dataset_file, column_name,device,key,n,vocab_size,accelerator,output_dir="./",name=""):
        super().__init__(model, tokenizer, max_new_tokens,dataset_file, column_name,device,accelerator,output_dir,name)
        self.key=key
        rng = mersenne_rng(key)
        self.n = n
        self.vocab_size = vocab_size
        self.xi = torch.tensor([rng.rand() for _ in range(self.n*self.vocab_size)]).view(self.n,self.vocab_size)
        self.shift = torch.randint(self.n, (1,))
     
        self.device = device
    def custom_parameters(self):
        return {
            **super().custom_parameters(),
            "llm_model":self.tokenizer.name_or_path,
            "dataset_file":self.dataset_file,
            "column_name":self.column_name,
            "key":self.key,
            "n":self.n,
            "max_new_tokens":self.max_new_tokens
        }
    def convert_data_to_format(self, dd):
        if "gold_completion" in dd:
            prefix = dd['prefix']
            gold_completion = dd['gold_completion']
        else:
            raise NotImplementedError("Subclasses should implement this method.")
        return prefix,gold_completion
    def process(self, prefix):
        inputs = self.tokenizer(prefix, return_tensors="pt", max_length=2048,padding=True,truncation=True).to(self.device)
        input_ids=inputs.input_ids
        attn=inputs.attention_mask
        def exp_sampling(probs,u):
             return torch.argmax(u ** (1/probs),axis=1).unsqueeze(-1)
        past = None
        for i in range(self.max_new_tokens):  
            with torch.no_grad():
                if past:
                    output = self.model(input_ids[:,-1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.model(input_ids)

            probs = torch.nn.functional.softmax(output.logits[:,-1, :self.vocab_size], dim=-1).cpu()
            token = exp_sampling(probs,self.xi[(self.shift+i)%self.n,:]).to(self.device)
            input_ids = torch.cat([input_ids, token], dim=-1)

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        return self.tokenizer.batch_decode(input_ids[:,len(inputs.input_ids[0]):], skip_special_tokens=True)
    
class GumbelStrategy(Strategy):
    name = "gumbel"
    def __init__(self,
                 model,
                 tokenizer,
                 dataset_file,
                 column_name,
                 device,
                 accelerator,
                 key,
                 ngram, 
                 max_new_tokens,
                 output_dir="./",
                 name="",
                 ):
        super().__init__(model, tokenizer,max_new_tokens, dataset_file, column_name, device, accelerator, output_dir, name, verbose=False)
        self.key=key
        self.ngram = ngram
        self.model =  GumbelSoftGeneratorNg(model=model, tokenizer=tokenizer, hash_key=self.key, ngram=self.ngram)
    def custom_parameters(self):
        return {
            **super().custom_parameters(),
            "llm_model":self.tokenizer.name_or_path,
            "dataset_file":self.dataset_file,
            "column_name":self.column_name,
            "max_new_tokens":self.max_new_tokens,
            "ngram":self.ngram,
            "key":self.key,
            "strategy":"gumbel",
            "type":"generate"
        }
    def process(self, text):
        _, full_output_list = self.model.generate(text, max_gen_len=self.max_new_tokens, top_p=0.95)
        return full_output_list

class RedGreenStrategy(Strategy):
    name = "red_green"

    def __init__(self,
                 model,
                 tokenizer,
                 dataset_file,
                 column_name,
                 device,
                 accelerator,
                 key,
                 gamma, 
                 delta,
                 max_new_tokens,
                 batch_size,
                 output_dir="./",
                 name="",
                 ):
        super().__init__(model, tokenizer,max_new_tokens, dataset_file, column_name, device, accelerator, output_dir, name, verbose=False)
        vocab = list(tokenizer.get_vocab().values())
        self.gamma= gamma
        self.delta=delta
        self.key=key
        self.watermark_processor = WatermarkLogitsProcessor(
            vocab=vocab,
            gamma=gamma,
            delta=delta,
            seeding_scheme=key,
            )
    def custom_parameters(self):
        return {
            **super().custom_parameters(),
            "llm_model":self.tokenizer.name_or_path,
            "dataset_file":self.dataset_file,
            "column_name":self.column_name,
            "max_new_tokens":self.max_new_tokens,
            "gamma":self.gamma,
            "delta":self.delta,
            "key":self.key,
            "device":str(self.device),
            "strategy":"red_green",
            "type":"generate"
        }
    def process(self, text):
        inputs = self.tokenizer(text, return_tensors="pt",padding=True).to(self.device)
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
                **inputs,
                max_length=len(inputs[0]) + self.max_new_tokens,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                logits_processor=LogitsProcessorList([self.watermark_processor]),
            )
        return self.tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)


class NoWaterMarkStrategy(Strategy):
    name = "red_green"

    def __init__(self,
                 model,
                 tokenizer,
                 dataset_file,
                 column_name,
                 device,
                 accelerator,
                 max_new_tokens,
                 output_dir="./",
                 name="",
                 ):
        super().__init__(model, tokenizer,max_new_tokens, dataset_file, column_name, device, accelerator, output_dir, name, verbose=False)

    def custom_parameters(self):
        return {
            **super().custom_parameters(),
            "llm_model":self.tokenizer.name_or_path,
            "dataset_file":self.dataset_file,
            "column_name":self.column_name,
            "max_new_tokens":self.max_new_tokens,
            "strategy":"no_watermark",
            "type":"generate"
        }
    def process(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
                **inputs,
                max_length=len(inputs[0]) + self.max_new_tokens,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
            )
        return self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()


def run(args):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    import torch
    from accelerate import Accelerator
    accelerator = Accelerator()
    print(accelerator.device)
    model_id = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        add_bos_token=True,
        add_eos_token=False,
        padding_side="left",
        use_auth_token=True,
    )
    tokenizer.add_special_tokens({'pad_token': '<padding_token>'})

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
        device_map=accelerator.device,
    )
    batch_size = args.batch_size
    print("method:", args.method)
    if args.method=="red_green":
        s = RedGreenStrategy(base_model
                        ,tokenizer,
                            args.dataset_file,
                            args.column_name,
                            base_model.device,
                            accelerator,
                            "ff-anchored_minhash_prf-2-True-111",
                            0.5,
                            4.0,
                            args.max_new_tokens,
                            batch_size,
                            args.output_dir,
                            args.name)
    elif args.method=="gumbel":
        s = GumbelStrategy(base_model
                        ,tokenizer,
                            args.dataset_file,
                            args.column_name,
                            base_model.device,
                            accelerator,
                            1234,
                            5,
                            args.max_new_tokens,
                            args.output_dir,
                            args.name)
    elif args.method=="no_watermarking":
        s = NoWaterMarkStrategy(base_model
                            ,tokenizer,
                            args.dataset_file,
                            args.column_name,
                            base_model.device,
                            accelerator,
                            args.max_new_tokens,
                            args.output_dir,
                            args.name)
    else:
        raise ValueError(f"Invalid method: {args.method}")
    s.execute(batch_size=batch_size)
    del base_model
    del tokenizer
    del accelerator
    return s.filename

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_id",type=str,default="meta-llama/Meta-Llama-3-8B")
    args.add_argument("--dataset_file",type=str,default="./datasets/wiki_inputs.jsonl")
    args.add_argument("--column_name",type=str,default="gold_completion")
    args.add_argument("--batch_size",type=int,default=16)
    args.add_argument("--max_new_tokens",type=int,default=300)
    args.add_argument("--output_dir",type=str,default="./final_experiments/generate")
    args.add_argument("--name",type=str,default="wiki")
    args.add_argument("--method",type=str,default="gumbel")
    args = args.parse_args()
    run(args)
