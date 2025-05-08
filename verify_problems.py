import argparse
import random
import sys
from execution import multi_execute_transformation, multi_execute_input_generator, execute_transformation
import numpy as np
import os
import re
import tqdm
import time
from utils import get_concepts_from_lines, get_description_from_lines
import subprocess
import json

class Problem:
    def __init__(self, source_code):
        self.source = source_code
        self.examples = []
        self.seeds = []
        self.source_uid = ""
        

    def add_example(self, input_grid, output_grid):
        self.examples.append((input_grid, output_grid))
    def to_dict(self):
        return {
            "source": self.source,
            "examples": [(input_grid.tolist(), output_grid.tolist()) for input_grid, output_grid in self.examples],
            "seeds": self.seeds,
            "source_uid": self.source_uid
        }


def check_grid(grid):
    # check the grid is well-formed, 2d numpy array of integers between 0-9
    try:
        assert isinstance(grid, np.ndarray)
        assert len(grid.shape) == 2
        assert grid.shape[0] > 0 and grid.shape[1] > 0
        # integer type
        assert np.all(np.equal(np.mod(grid, 1), 0))
        assert np.all((0 <= grid) & (grid <= 9))
    except AssertionError:
        return False
    except Exception as e:
        print(e)
        return False
    return True

def check_grids_all_equal(grids):
    """check a set of grids are all equal"""
    assert len(grids) > 0
    return all(np.array_equal(grids[0], grid) for grid in grids)

def check_diversity(grids, threshold):
    """check a set of grids is diverse, i.e. the grids should be sufficiently different from each other"""
    # TODO
    pass

def check_identity(input_grid, output_grid):
    try:
        assert check_grid(input_grid)
        assert check_grid(output_grid)
    except:
        breakpoint()
    return np.array_equal(input_grid, output_grid)

def generate_input_grids(problem_source, num_returns=3, timeout=1, function_name="generate_input", retries=20, deduplicate=True):
    """
    given a problem source code, generate an input grid
    """
    random.seed(0)

    return_input_grids = []
    tries = 0
    BATCH_SIZE = num_returns
    stats = { "non_well_formed_input": 0, "duplicate_input": 0 }
    while len(return_input_grids) < num_returns and tries < retries:
        random_seeds = [random.randint(0, 1<<30) for _ in range(BATCH_SIZE)]
        input_grids = multi_execute_input_generator([problem_source] * BATCH_SIZE, random_seeds, timeout, function_name)
        for input_grid in input_grids:
            if not check_grid(input_grid):
                tries += 1
                print('Non well-formed input grid')
                stats["non_well_formed_input"] += 1
                continue
            if deduplicate and any(np.array_equal(input_grid, existing_grid) for existing_grid in return_input_grids):
                tries += 1
                print('Duplicate input grid')
                stats["duplicate_input"] += 1
                continue
            return_input_grids.append(input_grid)

    return return_input_grids, stats

def get_random_color_mapping(only_non_black=True, permute_colors=None):
    """
    Get a random color mapping from 0-9 to 0-9 where 0 is black.
    If only_non_black is True, map 1-9 to 1-9.
    """
    if permute_colors is None:
        permute_colors = list(range(10))
    else:
        permute_colors = sorted(permute_colors)

    if only_non_black:
        if 0 in permute_colors:
            permute_colors.remove(0)

    shuffled_colors = list(permute_colors)
    random.shuffle(shuffled_colors)
    color_mapping = dict(zip(permute_colors, shuffled_colors))
    # add the rest of the colors as identity mapping
    for i in range(10):
        if i not in color_mapping:
            color_mapping[i] = i
    
    return color_mapping

def apply_color_mapping(grid, color_mapping):
    """
    Apply a color mapping to the grid
    """
    return np.vectorize(color_mapping.get)(grid)

def add_color_changing_code(problem_source, color_mapping=None):
    if color_mapping is None:
        color_mapping = {i: i for i in range(10)}
    color_code = f"""
Color.BLACK = {color_mapping[0]}
Color.BLUE = {color_mapping[1]}
Color.RED = {color_mapping[2]}
Color.GREEN = {color_mapping[3]}
Color.YELLOW = {color_mapping[4]}
Color.GREY = {color_mapping[5]}
Color.GRAY = {color_mapping[5]}
Color.PINK = {color_mapping[6]}
Color.ORANGE = {color_mapping[7]}
Color.TEAL = {color_mapping[8]}
Color.MAROON = {color_mapping[9]}
"""
    # Split the source code into lines
    lines = problem_source.split('\n')

    # Find the last line of the imports
    import_end_index = 0
    for i, line in enumerate(lines):
        if line.startswith("import") or line.startswith("from"):
            import_end_index = i + 1

    # Insert the color_code after the imports
    lines.insert(import_end_index, color_code)

    # Join the lines back into a single string
    modified_source = '\n'.join(lines)

    return modified_source

def run_transformation(source, input_grid, timeout=1, function_name="main", num_returns=50):
    """
    run the transformation on the input grid and return the output grid multiple times
    """
    random.seed(0)
    random_seeds = [random.randint(0, 1<<30) for _ in range(num_returns)]
    output_grids = multi_execute_transformation([source] * num_returns, [input_grid] * num_returns, random_seeds, timeout, function_name)
    return output_grids

def generate_solution(problem_source_uid, problem_source, examples, num_deterministic_check=20, timeout=1):
    """
    Generates output grids for each input grid (using the transformation) and checks them against the expected result.
    Return stats for the number of correct, incorrect, and unknown
    """
    start = time.time()
    problem = Problem(problem_source)

    stats = { "correct": 0, "incorrect": 0, "unknown": 0 }

    good_examples = []
    for example in examples:
        input_grid, output_grid = example["input"], example["output"]
        output_grids = run_transformation(problem_source, input_grid, timeout=timeout, num_returns=num_deterministic_check)
        correct = max([type(o) != str and output_grid == o.tolist() for o in output_grids])
        incorrect = max([type(o) == str or output_grid != o.tolist() for o in output_grids])
        if correct and not incorrect:
            stats["correct"] += 1
            good_examples.append(example)
        elif incorrect and not correct:
            stats["incorrect"] += 1
        else: stats["unknown"] += 1
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timeout", type=int, default=30, help="The total timeout value for a problem generation run")
    parser.add_argument("--exampledir", type=str, help="Input directory for the generated examples")
    args = parser.parse_args()

    total_timeout = args.total_timeout 

    problems_source = []
    problems_seeds = []
    problem_source_uids = []
    seeds = os.listdir("seeds")
    # filter files with .py extension and 8 hex value characters in the file name
    pattern = r"([0-9a-f]{8})\.py"
    problem_source_uids = [re.match(pattern, filename).group(1) for filename in seeds if re.match(pattern, filename)]
    # Now `matched_files` contains all the filenames that match the pattern

    if problem_source_uids:
        for problem_source_uid in problem_source_uids:
            with open(f"seeds/{problem_source_uid}.py") as f:
                source = f.read()
            problems_source.append(source)

    # For these UIDs, BARC fails on one or more ARC-AGI-1 example pairs.
    bad_uids = ["1b2d62fb", "25ff71a9", "28e73c20", "3428a4f5", "bbc9ae5d", "cf98881b",
                "feca6190", "25d8a9c8", "6fa7a44f", "995c5fa3", "9af7a82c", "db93a21d",
                "e48d4e1a", "f8b3ba0a", "017c7c7b", "0520fde7", "178fcbfb", "1caeab9d",
                "1fad071e", "2dee498d", "3618c87e", "3e980e27", "3f7978a0", "444801d8",
                "54d82841", "6d58a25d", "7447852a", "834ec97d", "8403a5d5", "8d5021e8",
                "8e5a5113", "9f236235", "a3df8b1e", "a79310a0", "a9f96cdd", "bd4472b8",
                "d4a91cb9", "e179c5f4", "fcc82909", "ff28f65a", "025d127b", "1bfc4729",
                "3ac3eb23", "8d510a79", "aabf363d", "d06dbe63", "d9f24cd1", "db3e9e38",
                "eb281b96", "8a004b2b", "29c11459", "caa06a1f"]
    overall_stats = { "non_deterministic": 0, "non_color_invariant": {"transformation_fail": 0, "non_well_formed": 0, "non_color_invariant": 0}, "identity": 0, "non_well_formed_output": 0, "black_output": 0, "timeout": 0, "non_well_formed_input": 0, "duplicate_input": 0, "total": 0}
    problems = []
    # failed_problems = []
    for i, problem_source in enumerate(problems_source if args.exampledir else tqdm.tqdm(problems_source)):
        problem_source_uid = problem_source_uids[i]
        examples = {}
        with open(f"{args.exampledir}/{problem_source_uid}.json") as f:
            import json
            problem_examples = json.loads(f.read())
            if type(problem_examples) != list:
                problem_examples = problem_examples["train"] + problem_examples["test"]
            examples[problem_source_uid] = problem_examples
        if not isinstance(problem_source, list):
            problem_source = [problem_source]
        for j, source in enumerate(problem_source):
            if args.exampledir:
                if problem_source_uid in bad_uids: continue
                if problem_source_uid not in examples: continue
                solution_stats = generate_solution(problem_source_uid, source, examples[problem_source_uid])
                print(problem_source_uid + ": " + str(solution_stats))

    print(f"Overall stats: {overall_stats}")


if __name__ == "__main__":
    main()
