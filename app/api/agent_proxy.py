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
search_code(code_str: str)
search_ast_pattern(pattern_code: str)
search_docstrings(search_text: str)
search_import(module_name: str)
search_variable(variable_name: str)
search_function_calls(function_name: str)

Provide your answer in JSON structure like this. Make sure none of the paths are argument placeholders like path/to/file, but are real paths.

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

Provide your answer in JSON structure like this. Make sure none of the paths are argument placeholders like path/to/file, but are real paths.

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

RETRY_PROXY_PROMPT = """
You are a helpful assistant tasked with extracting API calls and bug locations from a provided text. Your output must be a valid JSON structure.

**Text Structure:**

1. **Do we need more context?**  
   *(Extract API calls from this part. If none, leave the "API_calls" list empty.)*

2. **Where are bug locations?**  
   *(Extract bug locations from this part. If none, leave the "bug_locations" list empty.)*

**Instructions:**

- **API Calls:**
  - Extract any API calls mentioned in part 1.
  - Format each API call as a valid Python expression without argument placeholders.
    - **Correct:** `search_code("example_code")`
    - **Incorrect:** `search_code(code_str="example_code")` or `search_code(code_str)`
  - Ensure all arguments are specific and real (no placeholders like `"path/to/file"`).

- **Bug Locations:**
  - Extract bug locations mentioned in part 2.
  - Each bug location should be a dictionary with the following keys:
    - `"file"`: The real file path (omit if not specified).
    - `"class"`: The class name (include if specified).
    - `"method"`: The method name (include if specified).
  - A bug location must have at least one of `"class"` or `"method"`.

**Expected JSON Output:**

```json
{
    "API_calls": ["function1(argument1)", "function2(argument2)", ...],
    "bug_locations": [
        {"file": "file_path", "class": "class_name", "method": "method_name"},
        {"class": "class_name", "method": "method_name"},
        ...
    ]
}
```

**Example:**

If the text contains:

- **Part 1:** "We might need to use `search_method('my_method')` and `search_class('MyClass')`."
- **Part 2:** "The bug is located in `src/utils.py` within the `Helper` class."

Then the JSON output should be:

```json
{
    "API_calls": ["search_method('my_method')", "search_class('MyClass')"],
    "bug_locations": [
        {"file": "src/utils.py", "class": "Helper"}
    ]
}
```

**Additional Notes:**

- Ensure the final JSON output is valid and properly formatted.
- Do not include any argument placeholders or undefined variables.
- Use actual values provided in the text.
"""

RETRY_PROXY_PROMPT_POSTCOND = """
You are a helpful assistant tasked with extracting **API calls**, **bug locations**, and **postconditions** from a provided text. Your output must be formatted as valid JSON.

**Text Structure:**

1. **Do we need more context?**  
   *(Extract API calls from this part. If none, leave the `"API_calls"` list empty.)*

2. **Where are bug locations?**  
   *(Extract bug locations from this part. If none, leave the `"bug_locations"` list empty.)*

3. **What are the postconditions that need to be considered?**  
   *(Extract postconditions from this part. If none, leave the `"postconditions"` list empty.)*

---

**Instructions:**

### **API Calls:**

- Extract any API calls mentioned in **Part 1**.
- The available API calls are:

  - `search_method_in_class(method_name, class_name)`
  - `search_method_in_file(method_name, file_path)`
  - `search_method(method_name)`
  - `search_class_in_file(class_name, file_name)`
  - `search_class(class_name)`
  - `search_code_in_file(code_str, file_path)`
  - `search_code(code_str)`
  - `search_ast_pattern(pattern_code)`
  - `search_docstrings(search_text)`
  - `search_import(module_name)`
  - `search_variable(variable_name)`
  - `search_function_calls(function_name)`

- Format each API call as a valid Python expression **without argument placeholders**.
  - **Correct:** `search_code("example_code")`
  - **Incorrect:** `search_code(code_str="example_code")`, `search_code(code_str)`, or using placeholders like `"path/to/file"`
- Ensure all arguments are specific and real; do not use placeholders.

### **Bug Locations:**

- Extract bug locations mentioned in **Part 2**.
- Each bug location should be a dictionary with the following keys:

  - `"file"`: The real file path (omit if not specified).
  - `"class"`: The class name (include if specified).
  - `"method"`: The method name (include if specified).

- A bug location must have at least one of `"class"` or `"method"`.

### **Postconditions:**

- Extract postconditions mentioned in **Part 3**.
- Each postcondition should include both:

  - A **comment** explaining the postcondition (starting with `#`).
  - The actual **assert** statement.

- Combine the comment and assert statement into a **single string**, separated by a newline (`\n`).
- Each postcondition (comment + assert) should be a separate string in the list.

---

**Expected JSON Output:**

```json
{
    "API_calls": ["function1(argument1)", "function2(argument2)", ...],
    "bug_locations": [
        {"file": "file_path", "class": "class_name", "method": "method_name"},
        {"class": "class_name", "method": "method_name"},
        ...
    ],
    "postconditions": [
        "# Comment explaining the postcondition\nassert condition",
        "# Another comment\nassert another_condition",
        ...
    ]
}
```

---

**Example:**

If the text contains:

- **Part 1:**  
  "We might need to use `search_method('my_method')` and `search_class('MyClass')`."

- **Part 2:**  
  "The bug is located in `src/utils.py` within the `Helper` class."

- **Part 3:**  
  "Ensure that the result is not empty.  
  `# The result should not be empty`  
  `assert len(result) > 0`"

Then the JSON output should be:

```json
{
    "API_calls": ["search_method('my_method')", "search_class('MyClass')"],
    "bug_locations": [
        {"file": "src/utils.py", "class": "Helper"}
    ],
    "postconditions": [
        "# The result should not be empty\nassert len(result) > 0"
    ]
}
```

---

**Additional Notes:**

- **Validity:**
  - Ensure the final JSON output is valid and properly formatted.
  - Do not include any argument placeholders or undefined variables.
  - Use actual values provided in the text.

- **Formatting:**
  - Each postcondition should be a **single string** combining the comment and the assert statement, separated by a newline (`\n`).
  - Maintain the order of postconditions as they appear in the text.

- **Completeness:**
  - If any section (API calls, bug locations, or postconditions) has no relevant information, represent it with an empty list in the JSON output.
"""

def run_with_retries(text: str, retries=5) -> tuple[str | None, list[MessageThread]]:
    msg_threads = []
    for idx in range(1, retries + 1):
        logger.debug(
            "Trying to select search APIs in json where text is {}. Try {} of {}.",
            text,
            idx,
            retries,
        )

        res_text, new_thread = run(text)
        msg_threads.append(new_thread)

        extract_status, data = is_valid_json(res_text)

        if extract_status != ExtractStatus.IS_VALID_JSON:
            if idx == retries:
                logger.debug(
                    "Failed to extract json after {} retries with basic prompt. Trying another prompt",
                    retries,
                )
                msg_thread = MessageThread()
                msg_thread.add_system(RETRY_PROXY_PROMPT_POSTCOND if globals.enable_post_conditions else RETRY_PROXY_PROMPT)
                msg_thread.add_user(text)
                res_text, *_ = common.SELECTED_MODEL.call(
                    msg_thread.to_msg(), response_format="json_object"
                )
                extract_status, data = is_valid_json(res_text)
                if extract_status != ExtractStatus.IS_VALID_JSON:
                    logger.debug(
                        "Failed to extract json after {} retries with new prompt. Giving up",
                        retries,
                    )
                else:
                    msg_thread.add_model(res_text, [])  # no tools
                    msg_threads.append(msg_thread)
            else:

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