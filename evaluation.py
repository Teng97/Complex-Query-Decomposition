from sentence_transformers import SentenceTransformer
import json
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import evaluate
import pandas as pd
from comet import download_model, load_from_checkpoint

def compute_cos_sim(gt_ans, pred_ans, model):
    # Compute embeddings
    gt_embedding = model.encode(gt_ans)
    pred_embedding = model.encode(pred_ans)

    # Compute similarity
    return cosine_similarity([gt_embedding],[pred_embedding])[0][0]

def main_score():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HOTPOTQA", type=str, choices=['HOTPOTQA', 'LC-QuAD'],\
                        help="Which dataset use to evaluate the model.")
    parser.add_argument("--model", default="DecompRC", type=str, choices=['DecompRC', 'MuHeQA'],\
                        help="Which model to evaluate.")
    parser.add_argument("--type", default="bridge", type=str, choices=['bridge', 'intersection'],\
                        help="Which decomposition method use.")
    args = parser.parse_args()

    # Performance model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Comet model
    model_path = download_model("Unbabel/wmt22-comet-da")
    model_comet = load_from_checkpoint(model_path)
    
    rouge = evaluate.load('rouge')

    # Path of predicted answer
    pred_path = "/path"

    # Path of orignal answer
    gt_path = "/path"

    # Create pandas dataset to save results
    dataset = pd.DataFrame()

    # Open gt file
    with open(gt_path) as f:
        gt = json.load(f)
    print('Loaded gt!')
    f.close()

    gt_dict = {}

    # Extract the gt data to be accesible by ids
    for q in gt['data']:
        obj = q['paragraphs'][0]['qas'][0]
        gt_dict[obj['id']] = {'final_answers':obj['final_answers'], 'question':obj['question']}

    # Open predicted answers file
    with open(pred_path) as f:
        pred_ans_dict = json.load(f)
    print('Loaded predictions!')
    f.close()

    results = {}

    print('Computing metrics ...')
    # Select the ids from predicted answers
    for id in tqdm(pred_ans_dict):
        top_ans = None
        top_gt_ans = None
        top_cos_sim = None
        cos_sim = 0.0
        gt_p = gt_dict[id]
        # Select the predicted answers
        for pred_ans in pred_ans_dict[id]: 
            # Select the gt answers
            if not gt_p['final_answers']:  
                top_cos_sim = cos_sim
                top_ans = pred_ans
                top_gt_ans = "No answers"
            else:
                # Compute de cosine similarity and keep the highest 
                for gt_ans in gt_p['final_answers']:

                    pred_ans_text = str(pred_ans['text'])
                    gt_ans = str(gt_ans)

                    cos_sim = compute_cos_sim(gt_ans, pred_ans_text, model)

                    if top_cos_sim is None or cos_sim > top_cos_sim:
                        top_cos_sim = cos_sim
                        top_ans = pred_ans
                        top_gt_ans = gt_ans
        
        # Compute the other metrics from the qa with highest cosine similarity
        if top_ans is None:
            results[id] = {'multi_q': gt_p['question'], 'sq1': "Error", 'ans1': "Error", 'sq2': "Error", 'ans2': "Error",\
            'multi_gt':top_gt_ans, 'multi_pred': "Error", 'comet':0.0, 'cos_sim': 0.0, 'bleu': 0.0, 'rouge': 0.0, 'meteor': 0.0} 
        else:

            top_ans_text = str(top_ans['text'])
            top_gt_ans = str(top_gt_ans)

            # Compute cosine similarity
            cos_sim_sc = compute_cos_sim(top_gt_ans, top_ans_text, model)
            # Compute bleu
            bleu_sc = sentence_bleu([top_gt_ans.split()], top_ans_text.split(), weights=[(1)])
            # Compute rouge
            score_rouge = rouge.compute(predictions=[top_ans_text],references=[top_gt_ans])
            rouge_sc = score_rouge['rouge1']
            # Compute meteor
            meteor_sc = meteor_score([top_gt_ans.split()], top_ans_text.split())
            
            # Compute comet
            data = [
                {
                    "src": "",
                    "mt": top_ans_text,
                    "ref": top_gt_ans
                }
            ]
            score_comet = model_comet.predict(data, batch_size=1, gpus=1, progress_bar=False)
            comet_sc = score_comet['scores'][0]

            
            results[id] = {'multi_q': gt_p['question'], 'sq1': top_ans['q1'], 'ans1': top_ans['ans1'], 'sq2': top_ans['q2'], 'ans2': top_ans['ans2'],\
            'multi_gt':top_gt_ans, 'multi_pred': top_ans['text'], 'comet':comet_sc, 'cos_sim': cos_sim_sc, 'bleu': bleu_sc, 'rouge': rouge_sc, 'meteor': meteor_sc} 

    # Save the metrics
    dataset = pd.DataFrame.from_dict(results)
    dataset = (dataset.T)

    dataset.to_excel(f"results_metrics_+{args.dataset}+_sample_+{args.model}+_+{args.type}+.xlsx")

    print('Saved the computed metrics!')

if __name__ == '__main__':
    main_score()
