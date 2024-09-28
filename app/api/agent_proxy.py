"""
A proxy agent. Process raw response into json format.
"""

import inspect
from typing import Any

from loguru import logger

from app import globals
from app.data_structures import MessageThread
from app.model import common
from app.post_process import ExtractStatus, is_valid_json
from app.search.search_manage import SearchManager
from app.utils import parse_function_invocation

PROXY_PROMPT = """
You are a helpful assistant that retrieves API calls and bug locations from a text into json format.
The text will consist of two parts:
1. Do we need more context?
2. Where are bug locations?

Extract API calls from question 1 (leave empty if not exist) and bug locations from question 2 (leave empty if not exist).

The API calls include:
search_method_in_class(method_name: str, class_name: str)
search_method_in_file(method_name: str, file_path: str)
search_method(method_name: str)
search_class_in_file(self, class_name, file_name: str)
search_class(class_name: str)
search_code_in_file(code_str: str, file_path: str)
search_code(code_str: str)

Provide your answer in JSON structure like this:

{
    "API_calls": ["api_call_1(args)", "api_call_2(args)", ...],
    "bug_locations": [
        {"file": "path/to/file", "class": "class_name", "method": "method_name"},
        {"file": "path/to/file", "class": "class_name", "method": "method_name"},
        ...
    ]
}

NOTES:
- Ignore the argument placeholders in API calls. For example, search_code(code_str="str") should be search_code("str").
- Make sure each API call is written as a valid Python expression.
- A bug location should at least have a "class" or "method".
"""

PROXY_PROMPT_POSTCOND = """
You are a helpful assistant that retrieves API calls, bug locations, and postconditions from a text into json format.
The text will consist of three parts:
1. Do we need more context?
2. Where are bug locations?
3. What are the postconditions that need to be considered?

Extract API calls from question 1 (leave empty if not exist), bug locations from question 2 (leave empty if not exist), and postconditions from question 3 (leave empty if not exist).

The API calls include:
search_method_in_class(method_name: str, class_name: str)
search_method_in_file(method_name: str, file_path: str)
search_method(method_name: str)
search_class_in_file(self, class_name, file_name: str)
search_class(class_name: str)
search_code_in_file(code_str: str, file_path: str)
search_code(code_str: str)

Provide your answer in JSON structure like this:

{
    "API_calls": ["api_call_1(args)", "api_call_2(args)", ...],
    "bug_locations": [
        {"file": "path/to/file", "class": "class_name", "method": "method_name"},
        {"file": "path/to/file", "class": "class_name", "method": "method_name"},
        ...
    ],
    "postconditions": [
        "# Comment explaining the postcondition",
        "assert condition",
        "# Another comment",
        "assert another_condition",
        ...
    ]
}

NOTES:
- Ignore the argument placeholders in API calls. For example, search_code(code_str="str") should be search_code("str").
- Make sure each API call is written as a valid Python expression.
- A bug location should at least have a "class" or "method".
- Postconditions should include both comments (starting with #) explaining the condition and the actual assert statements.
- Each postcondition (comment + assert) should be a separate string in the list.
"""

def run_with_retries(text: str, retries=5) -> tuple[str | None, list[MessageThread]]:
    msg_threads = []
    for idx in range(1, retries + 1):
        logger.debug(
            "Trying to select search APIs in json. Try {} of {}.", idx, retries
        )

        res_text, new_thread = run(text)
        msg_threads.append(new_thread)

        extract_status, data = is_valid_json(res_text)

        if extract_status != ExtractStatus.IS_VALID_JSON:
            logger.debug("Invalid json. Will retry.")
            continue

        valid, diagnosis = is_valid_response(data)
        if not valid:
            logger.debug(f"{diagnosis}. Will retry.")
            continue

        logger.debug("Extracted a valid json")
        return res_text, msg_threads
    return None, msg_threads

def run(text: str) -> tuple[str, MessageThread]:
    """
    Run the agent to extract issue to json format.
    """

    msg_thread = MessageThread()
    prompt = PROXY_PROMPT_POSTCOND if globals.enable_post_conditions else PROXY_PROMPT
    msg_thread.add_system(prompt)
    msg_thread.add_user(text)
    res_text, *_ = common.SELECTED_MODEL.call(
        msg_thread.to_msg(), response_format="json_object"
    )

    msg_thread.add_model(res_text, [])  # no tools

    return res_text, msg_thread

def is_valid_response(data: Any) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Json is not a dict"

    has_api_calls = bool(data.get("API_calls"))
    has_bug_locations = bool(data.get("bug_locations"))
    has_postconditions = globals.enable_post_conditions and bool(data.get("postconditions"))

    if not (has_api_calls or has_bug_locations or (globals.enable_post_conditions and has_postconditions)):
        return False, "API_calls and bug_locations are empty" + (" and postconditions are empty" if globals.enable_post_conditions else "")

    if has_bug_locations:
        bug_locations = data["bug_locations"]
        if not isinstance(bug_locations, list):
            return False, "bug_locations must be a list"
        for loc in bug_locations:
            if not (loc.get("class") or loc.get("method")):
                return False, "Bug location not detailed enough"

    if has_api_calls:
        api_calls = data["API_calls"]
        if not isinstance(api_calls, list):
            return False, "API_calls must be a list"
        for api_call in api_calls:
            if not isinstance(api_call, str):
                return False, "Every API call must be a string"

            try:
                func_name, func_args = parse_function_invocation(api_call)
            except Exception:
                return False, "Every API call must be of form api_call(arg1, ..., argn)"

            function = getattr(SearchManager, func_name, None)
            if function is None:
                return False, f"the API call '{api_call}' calls a non-existent function"

            arg_spec = inspect.getfullargspec(function)
            arg_names = arg_spec.args[1:]  # first parameter is self

            if len(func_args) != len(arg_names):
                return False, f"the API call '{api_call}' has wrong number of arguments"

    if globals.enable_post_conditions and has_postconditions:
        postconditions = data["postconditions"]
        if not isinstance(postconditions, list):
            return False, "postconditions must be a list"
        if not postconditions:
            return False, "postconditions list is empty"
        for postcondition in postconditions:
            if not isinstance(postcondition, str):
                return False, "Every postcondition must be a string"

    return True, "OK"