"""
An agent, which is only responsible for the write_patch tool call.
"""

import json
import shutil
from collections.abc import Callable, Iterable
from copy import deepcopy
from os.path import join as pjoin
from pathlib import Path

from loguru import logger

from app import globals
from app.api import agent_common
from app.api.python import validation
from app.data_structures import MessageThread, MethodId
from app.log import print_acr, print_patch_generation
from app.model import common
from app.post_process import (
    ExtractStatus,
    extract_diff_one_instance,
    record_extract_status,
)
from app.task import SweTask, Task

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Your ultimate goal is to write a patch that resolves this issue.

When writing your patch:
1. Carefully analyze the issue and the provided context.
2. Ensure your solution addresses the specific problem described in the issue.
3. Consider potential side effects of your changes.
4. Verify that your complete solution resolves the issue without introducing new problems.

Remember, a successful patch fixes the immediate issue while maintaining the overall integrity of the system.
"""

SYSTEM_PROMPT_POSTCOND = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Your ultimate goal is to write a patch that resolves this issue while adhering to specific postconditions.

These postconditions define the expected behavior and constraints of the system after your patch is applied. They serve as a guide for your solution and a verification tool for its correctness.

When writing your patch:
1. Carefully analyze the provided postconditions before starting your implementation.
2. Use these postconditions as a guide to shape your solution strategy.
3. Ensure each part of your patch contributes to satisfying one or more postconditions.
4. After each modification, explicitly explain how it relates to or satisfies relevant postconditions.
5. If a modification doesn't directly relate to a postcondition, justify its necessity for the overall solution.
6. Consider potential edge cases or scenarios where the postconditions might be violated, and address them in your patch.
7. Verify that your complete solution collectively satisfies all provided postconditions.

Remember, a successful patch not only fixes the immediate issue but also maintains the system's integrity as defined by these postconditions. Your explanations of how the patch satisfies the postconditions are crucial for verifying the correctness of your solution.
"""


USER_PROMPT_INIT = """Write a list of fix locations, based on the retrieved context.\n\n
Return the list of locations in the format below.\n\nWithin `<file></file>`, replace `...` with actual file path. Within `<class></class>`, replace `...` with a class name if needed. Within `<method></method`, replace `...` with a method name.\n\n
You can write multiple locations if needed.

```
# candidate 1
<file>...</file>
<class>...</class>
<method>...</method>


# candidate 2
<file>...</file>
<class>...</class>
<method>...</method>

# candidate 3
...
```
"""

NEW_USER_PROMPT_INIT = """
**Task:**

Write a patch for the issue based on the retrieved context. Include all necessary code to fully solve the issue, including any required imports.

**Instructions:**

- **Format:** For each modification, follow the exact format provided below.
- **Placeholders:** Replace `...` with the appropriate content.
  - `<file>...</file>`: Replace `...` with the actual file path.
  - `<original>...</original>`: Replace `...` with the original code snippet from the program.
  - `<patched>...</patched>`: Replace `...` with the fixed version of the original code.
- **Code Blocks:**
  - Enclose each modification within triple backticks (```) immediately after the `# modification [number]` comment to signify a code block.
  - Do not include any other triple backticks in the response.
  - Ensure all code blocks and tags are properly closed.
- **Python Code:**
  - Pay attention to indentation, as the code is in Python.
  - Include necessary imports if required.
- **Multiple Modifications:**
  - You can write multiple modifications if needed, incrementing the modification number each time.

**Format for Each Modification:**

```
# modification [number]
```
```
<file>path/to/file.py</file>
<original>
[original code snippet]
</original>
<patched>
[patched code snippet]
</patched>
```

**Example:**

```
# modification 1
```
```
<file>src/utils.py</file>
<original>
def calculate_total(a, b):
    return a + b
</original>
<patched>
def calculate_total(a, b):
    return a + b + 1  # Fixed off-by-one error
</patched>
```

---

**Additional Notes:**

- **Consistency:** Ensure that the format is followed exactly to maintain consistency.
- **Clarity:** Clearly differentiate between the original and patched code within their respective tags.
- **Completeness:** All code necessary to solve the issue must be included in your answer.
- **No Extra Backticks:** Apart from the specified code blocks, do not include additional backticks in your response.
- **Closing Tags:** Whenever you open a block using a tag or triple backticks, make sure to close it properly.

---
"""

USER_PROMPT_INIT_POSTCOND = """Write a patch for the issue, based on the retrieved context and guided by the provided postconditions.

You can import necessary libraries.

Approach the task as follows:
1. Review the postconditions carefully before starting your implementation.
2. For each modification you make:
   a. Write the modification in the format specified below.
   b. Immediately after each modification, explain in detail how it relates to or satisfies the relevant postconditions.
   c. If a modification doesn't directly relate to a postcondition, explain why it's necessary for the overall solution.
3. After completing all modifications, provide a final summary explaining how your complete patch collectively satisfies all postconditions.

Return the patch in the format below. Within `<file></file>`, replace `...` with the actual file path. Within `<original></original>`, replace `...` with the original code snippet from the program. Within `<patched></patched>`, replace `...` with the fixed version of the original code. When adding original code and updated code, pay attention to indentation, as the code is in Python.

You can write multiple modifications if needed.

```
# modification 1
<file>...</file>
<original>...</original>
<patched>...</patched>
<postcondition_check>
Explain in detail how this modification satisfies or relates to the relevant postconditions.
If it doesn't directly relate to a postcondition, justify its necessity for the overall solution.
</postcondition_check>

# modification 2
<file>...</file>
<original>...</original>
<patched>...</patched>
<postcondition_check>
Explain in detail how this modification satisfies or relates to the relevant postconditions.
If it doesn't directly relate to a postcondition, justify its necessity for the overall solution.
</postcondition_check>

# modification 3
...

<final_postcondition_check>
Provide a comprehensive explanation of how your complete patch collectively satisfies all provided postconditions.
Address any potential edge cases or scenarios where the postconditions might be challenged by your implementation.
</final_postcondition_check>
```

Remember to consider all provided postconditions throughout your patch development process. Your explanations in the <postcondition_check> and <final_postcondition_check> sections are crucial for verifying the correctness of your solution.
"""

NEW_USER_PROMPT_INIT_POSTCOND = """
**Task:**

Write a patch for the issue based on the retrieved context and guided by the provided postconditions. Include all necessary code to fully solve the issue, including any required imports.

---

**Approach:**

1. **Review Postconditions:**

   - Carefully read and understand all the provided postconditions before starting your implementation.

2. **For Each Modification:**

   a. **Write the Modification:**

      - Follow the specified format for each modification.

   b. **Explain Relation to Postconditions:**

      - Immediately after each modification, provide a detailed explanation of how it satisfies or relates to the relevant postconditions.

   c. **Justify Necessity if Not Directly Related:**

      - If a modification doesn't directly relate to a postcondition, explain why it's necessary for the overall solution.

3. **Final Summary:**

   - After completing all modifications, provide a comprehensive explanation of how your complete patch collectively satisfies all postconditions.

---

**Formatting Instructions:**

- Return the patch using the exact format below.
- Replace `...` with the appropriate content.
- Pay attention to Python indentation in both the original and patched code.
- You can include multiple modifications as needed.

---

**Format for Each Modification:**

```
# modification [number]
<file>path/to/file.py</file>
<original>
[original code snippet]
</original>
<patched>
[patched code snippet]
</patched>
<postcondition_check>
[Detailed explanation of how this modification satisfies or relates to the relevant postconditions. If it doesn't directly relate, justify its necessity for the overall solution.]
</postcondition_check>
```

---

**Final Postcondition Check:**

After all modifications, include:

```
<final_postcondition_check>
[Comprehensive explanation of how your complete patch collectively satisfies all provided postconditions. Address any potential edge cases or scenarios where the postconditions might be challenged by your implementation.]
</final_postcondition_check>
```

---

**Example:**

```
# modification 1
<file>src/utils.py</file>
<original>
def calculate_total(a, b):
    return a + b
</original>
<patched>
def calculate_total(a, b):
    return a + b + 1  # Fixed off-by-one error
</patched>
<postcondition_check>
This modification corrects the calculation to account for the off-by-one error, satisfying Postcondition 2, which requires accurate total computation.
</postcondition_check>

# modification 2
<file>src/main.py</file>
<original>
if __name__ == "__main__":
    main()
</original>
<patched>
if __name__ == "__main__":
    initialize()
    main()
</patched>
<postcondition_check>
Adding the `initialize()` function ensures that the system state is set up before execution, which is necessary for the overall solution even though it's not directly related to a specific postcondition.
</postcondition_check>

<final_postcondition_check>
The complete patch ensures that all calculations are accurate and the system initializes correctly, collectively satisfying all provided postconditions and enhancing the program's reliability.
</final_postcondition_check>
```

---

**Additional Notes:**

- **Completeness:** Include all necessary code and imports required to fully implement the solution.
- **Clarity:** Provide detailed explanations in the `<postcondition_check>` and `<final_postcondition_check>` sections to verify the correctness of your solution.
- **Formatting:**

  - Use the exact tags and structure as specified.
  - Ensure all tags and code blocks are properly closed.
  - Maintain proper Python indentation.

- **No Extraneous Content:** Do not include any additional text outside of the specified format.
- **Review:** Double-check your modifications and explanations to ensure they align with the postconditions and fully address the issue.

---

**Remember:** Your explanations in the `<postcondition_check>` and `<final_postcondition_check>` sections are crucial for verifying the correctness of your solution. Consider all provided postconditions throughout your patch development process.
"""

def run_with_retries(
    message_thread: MessageThread,
    output_dir: str,
    task: Task,
    retries=3,
    print_callback: Callable[[dict], None] | None = None,
) -> tuple[str, float, int, int]:
    """
    Since the agent may not always write an applicable patch, we allow for retries.
    This is a wrapper around the actual run.
    """
    # (1) Get postconditions if enabled
    _, postconditions, _ = task.api_manager.generate_postconditions()  if globals.enable_post_conditions else []
   
    # (2) Select the appropriate system prompt and initial user prompt
    system_prompt = SYSTEM_PROMPT_POSTCOND if globals.enable_post_conditions else SYSTEM_PROMPT
    initial_user_prompt = USER_PROMPT_INIT_POSTCOND if globals.enable_post_conditions else USER_PROMPT_INIT
    
    # (3) Create a modified system prompt that includes postconditions if enabled
    modified_system_prompt = system_prompt
    if globals.enable_post_conditions and postconditions:
        postcondition_text = "\n".join(postconditions)
        modified_system_prompt += f"\n\nFor this specific issue, consider the following postconditions:\n{postcondition_text}\n\nEnsure your patch satisfies these postconditions while resolving the issue."

    # (4) Replace system prompt
    messages = deepcopy(message_thread.messages)
    new_thread: MessageThread = MessageThread(messages=messages)
    new_thread = agent_common.replace_system_prompt(new_thread, modified_system_prompt)

    # (5) Add the initial user prompt
    new_thread.add_user(initial_user_prompt)
    print_acr(initial_user_prompt, "patch generation", print_callback=print_callback)

    # (6) Add postconditions to the message thread (for reference) if enabled
    if globals.enable_post_conditions and postconditions:
        postcondition_prompt = "Remember to consider these postconditions when writing your patch:\n" + "\n".join(postconditions)
        new_thread.add_user(postcondition_prompt)
        print_acr(postcondition_prompt, "postconditions", print_callback=print_callback)

    can_stop = False
    result_msg = ""

    RETRY_COUNT = 5
    for i in range(1, retries + RETRY_COUNT):
        if i > 1:
            debug_file = pjoin(output_dir, f"debug_agent_write_patch_{i - 1}.json")
            with open(debug_file, "w") as f:
                json.dump(new_thread.to_msg(), f, indent=4)

        if can_stop or i > retries:
            break

        logger.info(f"Trying to write a patch. Try {i} of {retries}.")

        raw_patch_file = pjoin(output_dir, f"agent_patch_raw_{i}")

        # actually calling model
        res_text, *_ = common.SELECTED_MODEL.call(new_thread.to_msg())

        new_thread.add_model(res_text, [])  # no tools

        logger.info(f"Raw patch produced in try {i}. Writing patch into file.")

        with open(raw_patch_file, "w") as f:
            f.write(res_text)

        print_patch_generation(
            res_text, f"try {i} / {retries}", print_callback=print_callback
        )

        # Attempt to extract a real patch from the raw patch
        diff_file = pjoin(output_dir, f"extracted_patch_{i}.diff")
        extract_status, extract_msg = extract_diff_one_instance(
            raw_patch_file, diff_file
        )

        # record the extract status. This is for classifying the task at the end of workflow
        record_extract_status(output_dir, extract_status)

        if extract_status == ExtractStatus.APPLICABLE_PATCH:
            patch_content = Path(diff_file).read_text()
            print_acr(
                f"```diff\n{patch_content}\n```",
                "extracted patch",
                print_callback=print_callback,
            )

            # patch generated is applicable and all edits are ok, so we can think about validation
            if globals.enable_validation:
                # if we have a patch extracted, apply it and validate

                patch_is_correct, err_message, log_file = task.validate(diff_file)
                shutil.move(log_file, pjoin(output_dir, f"run_test_suite_{i}.log"))

                if patch_is_correct:
                    result_msg = (
                        "Written a patch that resolves the issue. Congratulations!"
                    )
                    new_thread.add_user(result_msg)  # just for logging
                    print_acr(
                        result_msg,
                        f"patch generation try {i} / {retries}",
                        print_callback=print_callback,
                    )
                    can_stop = True
                # the following two branches cannot be swapped, because
                # --enable-perfect-angelic is meant to override --enable-angelic
                elif globals.enable_perfect_angelic:
                    if not isinstance(task, SweTask):
                        raise NotImplementedError(
                            f"Angelic debugging not implemented for {type(task).__name__}"
                        )

                    msg = (
                        f"Written an applicable patch, but it did not resolve the issue. Error message: {err_message}.",
                    )

                    incorrect_locations = validation.perfect_angelic_debug(
                        task.task_id, diff_file, task.project_path
                    )
                    angelic_msg = angelic_debugging_message(incorrect_locations[0])

                    result_msg = f"{msg}\n{angelic_msg}"
                    new_thread.add_user(result_msg)
                    print_acr(
                        result_msg,
                        f"patch generation try {i} / {retries}",
                        print_callback=print_callback,
                    )
                    continue
                elif globals.enable_angelic:
                    raise NotImplementedError(
                        "Angelic debugging has not been integrated"
                    )
                else:
                    result_msg = f"Written an applicable patch, but it did not resolve the issue. {err_message} "
                    result_msg += " Please try again."
                    new_thread.add_user(result_msg)
                    print_acr(
                        result_msg,
                        f"patch generation try {i} / {retries}",
                        print_callback=print_callback,
                    )
                    continue
            elif globals.enable_perfect_angelic:
                if not isinstance(task, SweTask):
                    raise NotImplementedError(
                        f"Angelic debugging not implemented for {type(task).__name__}"
                    )

                incorrect_locations = validation.perfect_angelic_debug(
                    task.task_id, diff_file, task.project_path
                )

                msg = "Extracted a patch."
                if angelic_msg := angelic_debugging_message(incorrect_locations[0]):
                    result_msg = f"{msg}\n{angelic_msg}"
                else:
                    result_msg = msg

                new_thread.add_user(result_msg)
                print_acr(
                    result_msg,
                    f"patch generation try {i} / {retries}",
                    print_callback=print_callback,
                )
                continue
            elif globals.enable_angelic:
                raise NotImplementedError("Angelic debugging has not been integrated")
            else:
                result_msg = "Extracted a patch. Since validation is disabled, you should validation the patch later on. Ending the workflow."
                new_thread.add_user(result_msg)  # just for logging
                print_acr(
                    result_msg,
                    f"patch generation try {i} / {retries}",
                    print_callback=print_callback,
                )
                can_stop = True

        else:
            # we don't have a valid patch
            new_prompt = (
                "Your edit could not be applied to the program. "
                + extract_msg
                + " Please try again."
            )
            new_thread.add_user(new_prompt)
            print_acr(
                new_prompt,
                f"patch generation try {i} / {retries}",
                print_callback=print_callback,
            )
            
            # Switch to NEW_USER_PROMPT_INIT_POSTCOND or NEW_USER_PROMPT_INIT for subsequent attempts
            if i == 1:
                if globals.enable_post_conditions:
                    new_prompt = NEW_USER_PROMPT_INIT_POSTCOND
                else:
                    new_prompt = NEW_USER_PROMPT_INIT
                
                new_thread.add_user(new_prompt)
                print_acr(new_prompt, "switching to new prompt", print_callback=print_callback)

    result_msg = "Failed to write a valid patch."
    return result_msg


def angelic_debugging_message(
    incorrect_locations: Iterable[tuple[str, MethodId]],
) -> str:
    msg = []

    if incorrect_locations:
        msg.append("The following methods should not have been changed:")
        msg.extend(
            f"    {filename}: {method_id!s}"
            for filename, method_id in incorrect_locations
        )

    return "\n".join(msg)