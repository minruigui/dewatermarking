from tqdm import tqdm
from transformers import AutoTokenizer
from dewatermarking.watermark.extended_watermark_processor import  WatermarkDetector
import json
import os
import logging
import numpy as np
from transformers import AutoTokenizer
from dewatermarking.mersenne import mersenne_rng
from numba import jit
import numpy as np
import math
from dewatermarking.commons import SaveMixin
import argparse
from dewatermarking.wm import GumbelSoftDetectorNg
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.debug(f"Directory '{dir_path}' created.")
    else:
        logger.debug(f"Directory '{dir_path}' already exists.")
        

@jit(nopython=True)
def levenshtein(x, y, gamma=0.0):
    """
    Compute a variant of the Levenshtein distance with a custom cost matrix.
    
    Parameters:
    - x: 1D list or array of integers.
    - y: 2D numpy array where `y[j-1, x[i-1]]` is used to compute the cost.
    - gamma: The gap penalty.

    Returns:
    - The computed distance value as a float.
    """
    n = len(x)
    m = len(y)

    # Initialize the (n+1) x (m+1) distance matrix
    A = np.zeros((n+1, m+1), dtype=np.float32)

    # Fill the distance matrix
    for i in range(0, n+1):
        for j in range(0, m+1):
            if i == 0:
                A[i][j] = j * gamma  # Fill the first row
            elif j == 0:
                A[i][j] = i * gamma  # Fill the first column
            else:
                # Calculate the cost using the provided y matrix
                cost = math.log(1 - y[j-1, x[i-1]])
                # Compute the minimum cost among insertion, deletion, and substitution
                A[i][j] = min(
                    A[i-1][j] + gamma,        # Deletion
                    A[i][j-1] + gamma,        # Insertion
                    A[i-1][j-1] + cost        # Substitution
                )

    # Return the bottom-right corner of the matrix
    return A[n][m]

def permutation_test(tokens,key,n,k,vocab_size,n_runs=100):
    rng = mersenne_rng(key)
    xi = np.array([rng.rand() for _ in range(n*vocab_size)], dtype=np.float32).reshape(n,vocab_size)
    test_result = detect(tokens,n,k,xi)

    p_val = 0
    for run in range(n_runs):
        xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
        null_result = detect(tokens,n,k,xi_alternative)

        # assuming lower test values indicate presence of watermark
        p_val += null_result <= test_result

    return (p_val+1.0)/(n_runs+1.0)


def detect(tokens,n,k,xi,gamma=0.0):
    m = len(tokens)
    n = len(xi)

    A = np.empty((m-(k-1),n))
    for i in range(m-(k-1)):
        for j in range(n):
            A[i][j] = levenshtein(tokens[i:i+k],xi[(j+np.arange(k))%n],gamma)

    return np.min(A)
# Strategy Interface
class DetectStrategy(SaveMixin):
    def __init__(self,tokenizer,data,output_dir,name):
        super().__init__(output_dir=output_dir, name=name, action_type="detect")
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(
                                                    tokenizer,
                                                    add_bos_token=True,
                                                    add_eos_token=False,
                                                    padding_side="left",
                                                    use_auth_token=True,
                                                    )
        self.output_dir = output_dir
        self.name = name
        self.data= data
    def execute(self):
        self.evaluate(list(map(lambda x:x["completion"],self.data)))
    def evaluate(self, dataset):
        raise NotImplementedError("EvaluationStrategy subclasses must implement the evaluate method.")

class KuditipudiDetectStrategy(DetectStrategy):
    def __init__(self, tokenizer, filename,output_dir, name, key, n):
        super().__init__(tokenizer,filename, output_dir, name)
        self.n = n
        self.key=key
    def custom_parameters(self):
        return {
            "key":self.key,
            "n":self.n,
            "llm_model":self.tokenizer.name_or_path,
            
        }
    def evaluate(self, dataset):
        pbar = tqdm(total=len(dataset), desc="detecting")
        outputs= []
        for idx,text in enumerate(dataset):
            tokens = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
            pval = permutation_test(tokens,self.key,self.n,len(tokens),len(self.tokenizer))
            outputs.append(pval)
            pbar.update(1)
            if (idx+1)%5==0:
                self.save_outputs(outputs)
            self.save_outputs(outputs)
class RedGreenListDetectStrategy(DetectStrategy):
    def __init__(self, tokenizer,data, parameters, output_dir, name):
        super().__init__(tokenizer,data, output_dir, name)
        vocab = list(self.tokenizer.get_vocab().values())
        self._parameters = parameters
        self.detector = WatermarkDetector(
            vocab=vocab,
            gamma=parameters["gamma"],
            seeding_scheme=parameters['key'],
            device=parameters['device'],
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
    def custom_parameters(self):
        return {
            "key":self._parameters["key"],
            "gamma":self._parameters["gamma"],
            "llm_model":self.tokenizer.name_or_path,
            "method":"green_red",
            "name":self.name
            
        }
    def evaluate(self, dataset):
        pbar = tqdm(total=len(dataset), desc="detecting")
        outputs= []
        for idx,text in enumerate(dataset):
            if text is None or text == "" or len(text) < 100:
                pbar.update(1)
                continue
            score = self.detector.detect(text)
            del score["z_score_at_T"]
            outputs.append(score)
            pbar.update(1)
            if (idx+1)%5==0:
                self.save_outputs(outputs)
            self.save_outputs(outputs)
class GumbelSoftDetectStrategy(DetectStrategy):
    def __init__(self, tokenizer,data, parameters, output_dir, name):
        super().__init__(tokenizer,data, output_dir, name)
        self.detector = GumbelSoftDetectorNg(tokenizer=self.tokenizer, hash_key=parameters['key'], ngram=parameters['ngram'])
        self._parameters = parameters

    def custom_parameters(self):
        return {
            "key":self._parameters["key"],
            "ngram":self._parameters["ngram"],
            "llm_model":self.tokenizer.name_or_path,
            "method":"gumbel",
            "name":self.name
        }
    def evaluate(self, dataset):
        pbar = tqdm(total=len(dataset), desc="detecting")
        outputs= []
        for idx,text in enumerate(dataset):
            if text is None or text == "" or len(text) < 100:
                continue
            score = self.detector.get_szp_by_t(text)
            outputs.append(score)
            pbar.update(1)
            if (idx+1)%5==0:
                self.save_outputs(outputs)
            self.save_outputs(outputs)
def run(args):
    path = args.path
    ty = args.type

    data = json.load(open(path,"r"))
    current_parameters = data["parameters"]
    if current_parameters['strategy'] == "no_watermark" and ty == "current":
        raise ValueError("Cannot detect watermark on rephrased text on no watermark strategy")
    previous_parameters = None
    if current_parameters['dataset_file'] and current_parameters['type'] == "rephrase":
        previous_parameters = json.load(open(current_parameters['dataset_file'],"r"))["parameters"]
    parameters = None

    if ty == "previous":
        parameters = previous_parameters
    elif ty == "current":
        parameters = current_parameters
    else:
        raise ValueError("Type must be either 'previous' or 'current'")
    if args.name:
        name = args.name
    else:
        #get filename from  path
        filename = path.split("/")[-1]
        name = f"{filename}_{parameters['strategy']}_{ty}"
    if parameters["strategy"] == "red_green":
        dectectStrategy = RedGreenListDetectStrategy(tokenizer="meta-llama/Meta-Llama-3-8B",
                                            data=data['data'],
                                            parameters=parameters,
                                            output_dir=args.output_dir,
                                            name=name)
    elif parameters["strategy"] == "gumbel":
        dectectStrategy = GumbelSoftDetectStrategy(tokenizer="meta-llama/Meta-Llama-3-8B",
                                            data=data['data'],
                                            parameters=parameters,
                                            output_dir=args.output_dir,
                                            name=name)
    else:
        raise ValueError("Strategy must be either 'red_green' or 'gumbel'")
    dectectStrategy.execute()
    return dectectStrategy.filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,required=True)
    parser.add_argument("--type",type=str,required=True)
    parser.add_argument("--output_dir",type=str,required=True)
    args = parser.parse_args()
    run(args)