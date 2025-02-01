from dewatermarking.pipeline import run
import json
import argparse
import os
if __name__ == "__main__":
    checkpoint_file = './checkpoint.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('--recalculate_p_value', action='store_true')
    args = parser.parse_args()
    if args.recalculate_p_value:
        print("Recalculating p value...")
        if os.path.exists(checkpoint_file):
            data = json.load(open(checkpoint_file))
            if "history" in data:
                history = data["history"]
                for key in list(history.keys()):
                    if "p value detect" in key:
                        del history[key]
        json.dump(data, open(checkpoint_file, 'w'))
    dataset_files = [
        ('wiki','./datasets/wiki_inputs.jsonl'),
        ('qa','./datasets/qa_inputs.jsonl')]
    run(dataset_files,'./output',checkpoint_file)