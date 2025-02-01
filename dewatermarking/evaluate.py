from abc import ABC, abstractmethod
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import json
from sentence_transformers import util


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, original_texts: List[str], texts: List[str]):
        pass

class EmbeddingSimilarityStrategy(EvaluationStrategy):
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1"):
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5',trust_remote_code=True)


    def evaluate(self,data_path: str):
        similarity_list = []
        

        with open(data_path, 'r') as f:
            data = json.load(f)
            for row in data['data']:
                prefix = row['prefix'].replace('### Original: ', '').replace('\n ### Rewrite:', '')
                completion = row['completion']
    
                embedding_1 = self.model.encode(prefix, convert_to_tensor=True)
                embedding_2 = self.model.encode(completion, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
                # print(similarity)
                similarity_list.append(similarity.item())
            del self.model
            return {'mean': np.mean(similarity_list), 'std': np.std(similarity_list)}



class WatermarkDetectionStrategy(EvaluationStrategy):
    def __init__(self):
       pass

    def evaluate(self, data_path: str):
        p_values = []
        with open(data_path, 'r') as f:
            data = json.load(f)
            method = data['parameters']['method']
            if method == 'gumbel':
                for item in data['data']:
                    if item[3] >200:
                        p_values.append(item[2])
            elif method == 'green_red':
                for item in data['data']:
                    if item['num_tokens_scored'] >200:
                        p_values.append(item['p_value'])
            else:
                raise ValueError(f"Unsupported method: {method}")
        # print the mean of p_values and std of p_values
        return {'mean': np.mean(p_values), 'std': np.std(p_values),'count':len(p_values),
                "p value <= 0.05": len([p for p in p_values if p < 0.05]),
                "p value <= 0.1": len([p for p in p_values if p < 0.1]),
                "p value <= 0.01": len([p for p in p_values if p < 0.01])}
