"""
Post-process the output of the inference workflow with improved patch assignment.
"""

import json
import os
import re
import shutil
import subprocess
import logging
from collections import defaultdict
from collections.abc import Mapping
from enum import Enum
from glob import glob
from os.path import join as pjoin
from shutil import move

from app import utils as apputils
from app.api.patch_utils import parse_edits
from app.model import common
import ast
import difflib

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def count_and_organize_tasks(
    task_list: list[str], task_list_name: str, task_exp_names: list[str], expr_dir: str
):
    """
    A helper for extract_diff_patches.
    Generate a message to log the number of tasks in one list.
    Also organizes tasks in this list to a new folder in the experiment directory.

    Args:
        - task_list: a list of task ids
        - task_list_name: name for this list (one of the categories)
        - task_exp_names: list of individual experiment result dir names
        - expr_dir: the overall experiment directory.

    Returns:
        - message, a string message to be written to log file.
    """
    total_num_tasks = len(task_exp_names)

    # (1) get the message ready
    message = f"Total number of tasks in {task_list_name}: {len(task_list)}/{total_num_tasks}.\n"
    for task in task_list:
        message += f"\t {task}\n"

    # (2) create a new dir and move the experiment results of these tasks there
    new_dir = pjoin(expr_dir, task_list_name)
    os.makedirs(new_dir, exist_ok=True)
    for task_exp_name in task_exp_names:
        if any([task_exp_name.startswith(x) for x in task_list]):
            # this expr dir belongs to a task in the list
            old_dir = pjoin(expr_dir, task_exp_name)
            shutil.move(old_dir, new_dir)

    return message


# track status of patch extraction
class ExtractStatus(str, Enum):
    APPLICABLE_PATCH = "APPLICABLE_PATCH"
    PARTIALLY_APPLICABLE_PATCH = "PARTIALLY_APPLICABLE_PATCH"
    MATCHED_BUT_EMPTY_ORIGIN = "MATCHED_BUT_EMPTY_ORIGIN"
    MATCHED_BUT_EMPTY_DIFF = "MATCHED_BUT_EMPTY_DIFF"
    RAW_PATCH_BUT_UNMATCHED = "RAW_PATCH_BUT_UNMATCHED"
    RAW_PATCH_BUT_UNPARSED = "RAW_PATCH_BUT_UNPARSED"
    NO_PATCH = "NO_PATCH"
    IS_VALID_JSON = "IS_VALID_JSON"
    NOT_VALID_JSON = "NOT_VALID_JSON"

    def __lt__(self, other):
        # order from min to max
        order = [
            self.NO_PATCH,
            self.RAW_PATCH_BUT_UNPARSED,
            self.RAW_PATCH_BUT_UNMATCHED,
            self.MATCHED_BUT_EMPTY_DIFF,
            self.MATCHED_BUT_EMPTY_ORIGIN,
            self.PARTIALLY_APPLICABLE_PATCH,
            self.APPLICABLE_PATCH,
        ]
        self_index = order.index(self)
        other_index = order.index(other)
        return self_index < other_index

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(self.value)

    def to_dir_name(self, expr_dir: str):
        return pjoin(expr_dir, self.value.lower())

    @staticmethod
    def max(statuses):
        return sorted(statuses)[-1]


def record_extract_status(individual_expr_dir: str, extract_status: ExtractStatus):
    """
    Write extract status to file, so that we can read it again when
    classifying patches
    """
    # there is 1-to-1 correspondence between agent_patch_raw and extract_status
    # FIXME: it might be better to record these status in memory so they can be easily managed.
    record_file = pjoin(individual_expr_dir, "extract_status.json")
    if not os.path.isfile(record_file):
        # record for the first time
        with open(record_file, "w") as f:
            json.dump({"extract_status": [extract_status]}, f, indent=4)
    else:
        with open(record_file) as f:
            record = json.load(f)
        record["extract_status"].append(extract_status)
        with open(record_file, "w") as f:
            json.dump(record, f, indent=4)


def read_extract_status(individual_expr_dir: str) -> tuple[ExtractStatus, int]:
    """
    Read extract status from file. If there are multiple status recorded, read the best one.
    Returns:
        - The best extract status
        - The index of the best status in the list of all statuses. (0-based)
    """
    # we should read from the all the record
    record_file = pjoin(individual_expr_dir, "extract_status.json")
    if not os.path.isfile(record_file):
        # if no status file is written, means that we did not even
        # reach the state of extracting patches
        return ExtractStatus.NO_PATCH, -1
    with open(record_file) as f:
        record = json.load(f)
    # convert string to enum type
    all_status = [ExtractStatus(s) for s in record["extract_status"]]

    best_status = ExtractStatus.max(all_status)
    best_idx = all_status.index(best_status)
    return best_status, best_idx


def get_final_patch_path(individual_expr_dir: str) -> str | None:
    """
    Get the final patch path from the individual experiment directory.
    If there are multiple extracted patches, need to figure out which one is the best based
    on the patch extraction history.
    """
    _, best_index = read_extract_status(individual_expr_dir)
    best_patch_name = f"extracted_patch_{best_index + 1}.diff"
    final_patch_path = pjoin(individual_expr_dir, best_patch_name)

    if not os.path.isfile(final_patch_path):
        return None

    return final_patch_path


def extract_diff_one_instance(
    raw_patch_file: str, extracted_file: str, standalone_mode: bool = False
) -> tuple[ExtractStatus, str]:
    """
    Extract .diff patches for one instance using advanced parsing techniques.
    Args:
        - raw_patch_file: Path to the raw patch file produced by model.
        - extracted_file: Path where the extracted diff file goes.
        - standalone_mode: If True, the function is called from the special --extract-patch mode.
                           Specify this to True if using this function as it is for testing.
    Returns:
        - ExtractStatus.
        - An additional string containing more explanation on how patch extraction failed.
          If everything is successful, this string is empty.
    """
    # (1) get the meta data for this task
    task_dir = os.path.dirname(raw_patch_file)
    meta_file = pjoin(task_dir, "meta.json")
    with open(meta_file) as f:
        meta = json.load(f)

    task_info = meta["task_info"]
    setup_info = meta["setup_info"]
    repo_path = setup_info["repo_path"]  # the project dir
    base_commit = task_info["base_commit"]  # the commit to checkout

    if not os.path.isfile(raw_patch_file):
        return ExtractStatus.NO_PATCH, "No raw patch file is found."

    with open(raw_patch_file) as f:
        patch_content = f.read()

    # (2) Try to parse the patch content as JSON first (if applicable)
    json_status, json_data = is_valid_json(patch_content)
    if json_status == ExtractStatus.IS_VALID_JSON:
        # Handle JSON patches if required
        pass  # Implement JSON-specific handling if necessary

    # (3) Advanced parsing using regex and AST
    try:
        edits = parse_edits_with_advanced_parsing(patch_content)
    except Exception as e:
        logger.error(f"Exception {e} happened when parsing edits.")
        return (
            ExtractStatus.RAW_PATCH_BUT_UNPARSED,
            f"Exception {e} happened when parsing edits.",
        )

    if not edits:
        logger.warning("No edits can be parsed.")
        return ExtractStatus.RAW_PATCH_BUT_UNPARSED, "No edits can be parsed."

    # (4) Apply the edits with improved heuristics
    with apputils.cd(repo_path):
        if standalone_mode:
            # in special --extract-patch mode
            apputils.repo_reset_and_clean_checkout(base_commit)
        else:
            # extracting patch in the write_patch loop
            # we should not reset to base commit, because previously we created a new commit
            # containing the test_patch content. We should just clean the changes until HEAD.
            apputils.repo_clean_changes()

        unmatched_edit_indexes = []
        partially_matched = False
        for idx, edit in enumerate(edits):
            target_file = edit.filename
            # Advanced file matching
            found_file = find_file_with_advanced_matching(repo_path, target_file)
            if found_file is None:
                logger.warning(f"File {target_file} not found in repository.")
                unmatched_edit_indexes.append(idx)
                continue
            # Try to apply this edit
            success = apply_edit_with_improved_matching(edit, found_file)
            if not success:
                logger.warning(
                    f"Failed to apply edit number {idx+1} for file {found_file}."
                )
                unmatched_edit_indexes.append(idx)
                continue
            else:
                # Check for partial matches
                if is_partial_match(edit, found_file):
                    partially_matched = True

        if len(unmatched_edit_indexes) == len(edits):
            # None of the edits can be matched
            apputils.repo_clean_changes()
            return (
                ExtractStatus.RAW_PATCH_BUT_UNMATCHED,
                "None of the edits can match the original program.",
            )

        # Prepare a message about unmatched edits
        if unmatched_edit_indexes:
            unmatched_msg = f"Edits number {','.join([str(x+1) for x in unmatched_edit_indexes])} cannot be matched to the original program."
        else:
            unmatched_msg = ""

        # Get the diff
        diff = apputils.run_command(
            ["git", "diff"], stdout=subprocess.PIPE
        ).stdout.decode()

        # Clean changes
        apputils.repo_clean_changes()

        if not diff:
            # Diff file is empty
            msg = (
                unmatched_msg
                + " The matched edits do not introduce any change to the codebase."
            )
            return ExtractStatus.MATCHED_BUT_EMPTY_DIFF, msg

        edits_with_empty_before = [
            str(idx + 1) for idx, edit in enumerate(edits) if not edit.before.strip()
        ]
        if edits_with_empty_before:
            numbers = ", ".join(edits_with_empty_before)
            msg = f"Please contain **non-whitespace** original code snippet in edits number {numbers}."
            return ExtractStatus.MATCHED_BUT_EMPTY_ORIGIN, msg

        # Save the diff
        with open(extracted_file, "w") as f:
            f.write(diff)

        # Determine if the patch is partially applicable
        if partially_matched or unmatched_edit_indexes:
            return ExtractStatus.PARTIALLY_APPLICABLE_PATCH, unmatched_msg

        # All edits applied successfully
        return ExtractStatus.APPLICABLE_PATCH, unmatched_msg


def parse_edits_with_advanced_parsing(patch_content: str):
    """
    Parse the patch content using advanced regex and AST parsing.
    """
    # Implement advanced parsing logic here
    # Enhanced regex patterns to handle flexible whitespace, indentation, and function definitions
    edits = []

    # Split the content into edits based on some delimiter or pattern
    edit_patterns = re.compile(r"Edit\s*\d+:", re.MULTILINE)
    raw_edits = edit_patterns.split(patch_content)

    for raw_edit in raw_edits:
        if not raw_edit.strip():
            continue
        # Use regex to extract filename, before, and after code blocks
        filename_match = re.search(r"File\s*:\s*(.*)", raw_edit)
        before_match = re.search(r"Before\s*:\s*(.*?)\nAfter\s*:", raw_edit, re.DOTALL)
        after_match = re.search(r"After\s*:\s*(.*)", raw_edit, re.DOTALL)

        if filename_match and before_match and after_match:
            filename = filename_match.group(1).strip()
            before = before_match.group(1).strip()
            after = after_match.group(1).strip()

            # Create an Edit object
            edit = Edit(filename=filename, before=before, after=after)
            edits.append(edit)
        else:
            # Log unmatched patterns for further analysis
            logger.debug(f"Unmatched edit pattern in raw edit: {raw_edit}")

    return edits


def find_file_with_advanced_matching(repo_path: str, target_file: str) -> str | None:
    """
    Use advanced matching to find the target file in the repository.
    """
    # Implement advanced file matching logic
    # For example, handle cases where the file path is partial or uses placeholders
    # Use regex to match file names with slight differences
    possible_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(os.path.basename(target_file)):
                possible_files.append(os.path.join(root, file))
    if len(possible_files) == 1:
        return possible_files[0]
    elif len(possible_files) > 1:
        # Use more sophisticated matching if multiple files are found
        for file in possible_files:
            if file == target_file:
                return file
    return None


def apply_edit_with_improved_matching(edit, found_file):
    """
    Apply the edit to the file using improved regex and AST parsing.
    """
    # Read the original file content
    with open(found_file, 'r') as f:
        original_content = f.read()

    # Flexible whitespace handling
    before_pattern = re.sub(r'\s+', r'\s+', re.escape(edit.before.strip()))
    before_regex = re.compile(before_pattern, re.MULTILINE)

    # Attempt to find the before code block in the original content
    match = before_regex.search(original_content)
    if not match:
        logger.debug(f"Original code block not found in {found_file} for edit.")
        return False

    # Prepare the after code block with proper indentation
    indent = ' ' * (len(match.group(0)) - len(match.group(0).lstrip(' ')))
    after_code = '\n'.join([indent + line for line in edit.after.strip().split('\n')])

    # Replace the before code block with the after code block
    patched_content = before_regex.sub(after_code, original_content)

    # Write the patched content back to the file
    with open(found_file, 'w') as f:
        f.write(patched_content)

    return True


def is_partial_match(edit, found_file):
    """
    Determine if the edit is a partial match.
    """
    # Implement logic to check for partial matches
    # For example, use difflib to compare the before and after snippets
    with open(found_file, 'r') as f:
        file_content = f.read()

    ratio_before = difflib.SequenceMatcher(None, edit.before, file_content).quick_ratio()
    ratio_after = difflib.SequenceMatcher(None, edit.after, file_content).quick_ratio()
    # Define a threshold for partial match
    THRESHOLD = 0.5
    return (ratio_before < 1.0 and ratio_before > THRESHOLD) or (ratio_after < 1.0 and ratio_after > THRESHOLD)


def organize_experiment_results(expr_dir: str):
    """
    Organize the experiment result directories into categories based on improved heuristics.
    """
    # (1) find all the task experiment directories
    task_exp_names = [
        x
        for x in os.listdir(expr_dir)
        if os.path.isdir(pjoin(expr_dir, x))
        and "__" in x  # for filtering out other dirs like "applicable_patch"
    ]
    task_exp_dirs = [pjoin(expr_dir, x) for x in task_exp_names]

    # Start organizing
    for extract_status in ExtractStatus:
        os.makedirs(extract_status.to_dir_name(expr_dir), exist_ok=True)

    for task_dir in task_exp_dirs:
        extract_status, _ = read_extract_status(task_dir)
        corresponding_dir = extract_status.to_dir_name(expr_dir)
        shutil.move(task_dir, corresponding_dir)


# NOTE: only used in the special mode of only extracting patches
def extract_diffs_and_organize_tasks(expr_dir: str):
    """
    Extract patches for all instances using improved parsing and organize them into categories.
    """
    log_file = pjoin(expr_dir, "extract_patches.log")
    log_file_handle = open(log_file, "w")
    task_exp_names = [
        x
        for x in os.listdir(expr_dir)
        if os.path.isdir(pjoin(expr_dir, x))
        and "__" in x  # for filtering out other dirs like "applicable_patch"
    ]
    task_exp_dirs = [pjoin(expr_dir, x) for x in task_exp_names]
    task_exp_dirs = sorted(task_exp_dirs)

    # Mapping from ExtractStatus to a list of task ids
    all_extract_stats: Mapping[ExtractStatus, list[str]] = defaultdict(list)

    # Work on each individual task directory
    for task_dir in task_exp_dirs:
        # (1) gather some information from the meta file
        meta_file = pjoin(task_dir, "meta.json")
        with open(meta_file) as f:
            meta = json.load(f)
        task_id = meta["task_id"]

        log_file_handle.write(f"\n\n\nExtracting patch for task {task_id}.\n")
        logger.info(f"Extracting patch for task {task_id}.")

        # (2) find the latest raw patch file
        raw_patch_files = [
            x for x in os.listdir(task_dir) if x.startswith("agent_patch_raw_")
        ]
        if not raw_patch_files:
            record_extract_status(task_dir, ExtractStatus.NO_PATCH)
            all_extract_stats[ExtractStatus.NO_PATCH].append(task_id)
            continue
        # find the most recent one
        numbers = [int(file.split("_")[-1]) for file in raw_patch_files]
        numbers.sort()
        if not numbers:
            record_extract_status(task_dir, ExtractStatus.NO_PATCH)
            all_extract_stats[ExtractStatus.NO_PATCH].append(task_id)
            continue

        all_status = []
        for num in numbers:
            raw_patch_file = pjoin(task_dir, f"agent_patch_raw_{num}")

            print(task_id, num)
            # (3) perform the actual extraction
            extracted_file = pjoin(task_dir, f"extracted_patch_{num}.diff")
            extract_status, _ = extract_diff_one_instance(
                raw_patch_file, extracted_file, standalone_mode=True
            )
            all_status.append(extract_status)

            record_extract_status(task_dir, extract_status)
            all_extract_stats[extract_status].append(task_id)

        log_file_handle.write(
            f"\tPatch extraction status: {ExtractStatus.max(all_status)}\n"
        )
        logger.info(
            f"Patch extraction status for task {task_id}: {ExtractStatus.max(all_status)}"
        )

    # Tasks have been categorized, now move them to specific folders based on the result
    organize_experiment_results(expr_dir)
    log_file_handle.close()


def extract_swe_bench_input(dir: str):
    """
    After diff format patch files have been extracted, collect them and write a single file for swe-bench.

    Returns:
        - path to swe-bench input file.
    """
    # Only look into applicable_patch and partially_applicable_patch dirs
    applicable_dirs = ['applicable_patch', 'partially_applicable_patch']
    all_results = []

    for status_dir in applicable_dirs:
        applicable_res_dir = pjoin(dir, status_dir)
        # Figure out what tasks have applicable patches
        if not os.path.exists(applicable_res_dir):
            continue
        task_dirs = [
            x
            for x in os.listdir(applicable_res_dir)
            if os.path.isdir(pjoin(applicable_res_dir, x))
        ]
        task_dirs = [pjoin(applicable_res_dir, x) for x in task_dirs]
        patch_files = [pjoin(x, "agent_patch_raw") for x in task_dirs]
        patch_files = [os.path.abspath(x) for x in patch_files]

        # Diff files have the name extracted_patch_{1,2,3...}.diff
        # We take the one with the largest index
        diff_files = []
        for x in task_dirs:
            extracted_patches = glob(pjoin(x, "extracted_patch_*.diff"))
            extracted_patches.sort(
                key=lambda name: int(name.removesuffix(".diff").split("_")[-1]),
                reverse=True,
            )
            diff_files.append(extracted_patches[0])

        diff_files = [os.path.abspath(x) for x in diff_files]
        patch_files = [x for x in patch_files if os.path.isfile(x)]
        diff_files = [x for x in diff_files if os.path.isfile(x)]

        for diff_file in diff_files:
            task_dir = os.path.dirname(diff_file)
            meta_file = pjoin(task_dir, "meta.json")
            with open(meta_file) as f:
                meta = json.load(f)
            task_id = meta["task_id"]
            this_result = {}
            this_result["instance_id"] = task_id
            this_result["model_name_or_path"] = common.SELECTED_MODEL.name
            with open(diff_file) as f:
                diff_content = f.read()
            if not diff_content:
                # empty diff file, don't bother sending it to swe-bench
                continue
            this_result["model_patch"] = diff_content
            all_results.append(this_result)

    swe_input_file = pjoin(dir, "predictions_for_swebench.json")
    with open(swe_input_file, "w") as f:
        json.dump(all_results, f, indent=4)

    return swe_input_file


def is_valid_json(json_str: str) -> tuple[ExtractStatus, list | dict | None]:
    """
    Check whether a json string is valid.
    """
    try:
        data = json.loads(json_str)
    except json.decoder.JSONDecodeError:
        return ExtractStatus.NOT_VALID_JSON, None
    return ExtractStatus.IS_VALID_JSON, data


"""
Main entries of the module.
"""


def reextract_organize_and_form_inputs(expr_dir: str):
    """
    Move individual experiment dirs out of the categories (applicable_patch, etc.),
    before extracting patches and organizing again.
    """
    abs_expr_dir = os.path.abspath(expr_dir)
    un_classify_expr_dir(abs_expr_dir)
    extract_diffs_and_organize_tasks(abs_expr_dir)


def un_classify_expr_dir(expr_dir: str):
    individual_expr_dirs = []
    for individual_expr_dir in glob(pjoin(expr_dir, "*", "*__*")):
        assert "info.log" in os.listdir(
            individual_expr_dir
        ), f"{individual_expr_dir} has no info.log"
        individual_expr_dirs.append(individual_expr_dir)

    for d in individual_expr_dirs:
        move(d, expr_dir)


def extract_organize_and_form_input(expr_dir):
    """
    From a directory of raw experiment result dirs, extract diff patches, organize them into
    categories, and form a single file that can be used by swe-bench.
    Args:
        - expr_dir: the overall experiment directory.
    """
    abs_expr_dir = os.path.abspath(expr_dir)
    extract_diffs_and_organize_tasks(abs_expr_dir)
    extract_swe_bench_input(abs_expr_dir)


def organize_and_form_input(expr_dir):
    """
    Only organize the experiment directories into a few categories.
    Args:
        - expr_dir: the overall experiment directory.
    """
    organize_experiment_results(expr_dir)
    swe_input_file = extract_swe_bench_input(expr_dir)
    return swe_input_file


# Define the Edit class used in parsing
class Edit:
    def __init__(self, filename: str, before: str, after: str):
        self.filename = filename
        self.before = before
        self.after = after
