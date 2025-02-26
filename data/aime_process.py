import argparse
import json
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs

# To extract the solution for each prompts in the dataset
# def extract_solution(solution_str):
# ...

def preprocess():
    # Load generated problems from jsonl file
    generated_results = []
    with open('generated_results_v3.jsonl', 'r') as f:
        for line in f:
            generated_results.append(json.loads(line))
    # Load all problems from jsonl file
    all_problems1 = []
    with open('all_problems_1.jsonl', 'r') as f:
        for line in f:
            all_problems1.append(json.loads(line))

    clean_problems = set()
    for problem in all_problems1:
        clean_problems.add((problem['type'], problem['year'], problem['number']))

    print('Filter: ', len(clean_problems), len(all_problems1))

    filtered_problems = []
    for problem in generated_results:
        if 'AMC' in problem['type'] and '2023' in problem['year']:
            print('filtered ', problem['type'], problem['year'], problem['number'])
            continue 
        if 'AIME' in problem['type'] and '2024' in problem['year']:
            print('filtered ', problem['type'], problem['year'], problem['number'])
            continue 
        if (problem['type'], problem['year'], problem['number']) in clean_problems:
            filtered_problems.append(problem)
    # Begin online ppo question processing:
    write_data_ppo = []
    for idx, problem in enumerate(filtered_problems):
        question = problem['question']
        correct_solutions = []
        wrong_solutions = []
        for s_idx, cor in enumerate(problem['corrects']):
            if cor:
                correct_solutions.append(problem['solutions'][s_idx])
            else:
                wrong_solutions.append(problem['solutions'][s_idx])
        if len(correct_solutions) == 0:
            continue
        # Rank solutions by (correctness, length) Tuple
        correct_solutions = sorted(correct_solutions, key=lambda x: len(x), reverse=True)
        wrong_solutions = sorted(wrong_solutions, key=lambda x: len(x), reverse=True)
        # Create DPO pairs
        max_len = len(correct_solutions[0])
        min_len = len(correct_solutions[-1])

        if len(wrong_solutions) == 0:
            wrong_max_len = max_len
            wrong_min_len = min_len
        else:
            wrong_max_len = max(max_len, len(wrong_solutions[0]))
            wrong_min_len = min(min_len, len(wrong_solutions[-1]))

        answer = problem["answer"]
        assert isinstance(question, str), question
        if not isinstance(answer, str):
            print(idx, answer)
            continue
        assert isinstance(answer, str), f"{idx, answer, problem['extracted_answers']}"
        assert isinstance(max_len, int), max_len
        assert isinstance(min_len, int), min_len
        write_data_ppo.append({
            "question": question,
            "answer": answer,
            "max_len": max_len,
            "min_len": min_len,
        })

    with open('math_ppo_question_v1.json', 'w') as f:
        json.dump(write_data_ppo, f, ensure_ascii=False, indent=2)

def make_map_fn(split):

    def process_fn(example, idx):
        data = {}
        question = example.pop("question")
        answer = example.pop("answer")
        data = {
            "prompt": [
                {"role": "system", "content": 'Please reason step by step, and put your final answer within \\boxed{{}}.'},
                {"role": "human", "content": question},
            ],
            "data_source": "amc-aime",
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                "split": split,
                "index": idx,
            }
        }
        return data

    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/workspace/verlpy310/data/aime')

    args = parser.parse_args()

    # preprocess()
    num_few_shot = 5
    data_source = '/workspace/verlpy310/data/math_ppo_question_v1.json'

    dataset = datasets.load_dataset("json", data_files=data_source, split="train")
    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
