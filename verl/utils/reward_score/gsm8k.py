# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import random

# def extract_solution(solution_str, method='strict'):
#     assert method in ['strict', 'flexible']

#     if method == 'strict':
#         # this also tests the formatting of the model
#         solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
#         if solution is None:
#             final_answer = None
#         else:
#             final_answer = solution.group(0)
#             final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
#     elif method == 'flexible':
#         answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
#         final_answer = None
#         if len(answer) == 0:
#             # no reward is there is no answer
#             pass
#         else:
#             invalid_str = ['', '.']
#             # find the last number that is not '.'
#             for final_answer in reversed(answer):
#                 if final_answer not in invalid_str:
#                     break
#     return final_answer
        
def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {ground_truth}")
        print(f"Extracted equation: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if do_print:
            print(f"No answer found")
        return 0
        
    # Evaluate
    try:
        if abs(result - ground_truth) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct answer: {answer} = {ground_truth}")
            return score
        else:
            if do_print:
                print(f"Wrong result: answer = {answer}, target = {ground_truth}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score 