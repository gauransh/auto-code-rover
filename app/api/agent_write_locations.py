"""
An agent, which is only responsible for the summarize locations tool call.
"""

import json
import re
from collections.abc import Callable
from copy import deepcopy
from os.path import join as pjoin
from pathlib import Path

from loguru import logger

from app.api import agent_common
from app.data_structures import MessageThread
from app.log import print_acr, print_fix_loc_generation
from app.model import common
from app.task import Task

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
You ultimate goal is to write a list of locations that you can give to another devloper.
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
    Since the agent may not always write a correct list, we allow for retries.
    This is a wrapper around the actual run.
    """
    # (1) replace system prompt
    messages = deepcopy(message_thread.messages)
    new_thread: MessageThread = MessageThread(messages=messages)
    new_thread = agent_common.replace_system_prompt(new_thread, SYSTEM_PROMPT)

    # (2) add the initial user prompt
    new_thread.add_user(USER_PROMPT_INIT)
    print_acr(
        USER_PROMPT_INIT, "fix location generation", print_callback=print_callback
    )

    

    can_stop = False
    result_msg = ""

    logger.info("Starting the agent to propose fix locations.")

    for i in range(1, retries + 2):
        if i > 1:
            debug_file = pjoin(output_dir, f"debug_agent_propose_locs_{i - 1}.json")
            with open(debug_file, "w") as f:
                json.dump(new_thread.to_msg(), f, indent=4)

        if can_stop or i > retries:
            break

        logger.info(f"Trying to propose fix locations. Try {i} of {retries}.")

        raw_location_file = pjoin(output_dir, f"agent_loc_list_{i}")

        # actually calling model
        res_text, *_ = common.SELECTED_MODEL.call(new_thread.to_msg())

        new_thread.add_model(res_text, [])  # no tools

        logger.info(f"Fix locations produced in try {i}. Writing locations into file.")

        with open(raw_location_file, "w") as f:
            f.write(res_text)

        # print("HI", raw_location_file)

        print_fix_loc_generation(
            res_text, f"try {i} / {retries}", print_callback=print_callback
        )

        fragment_pattern = re.compile(
            r"<file>(.*?)</file>\s*<class>(.*?)</class>\s*<method>(.*?)</method>"
        )

        result_msg = "No fragments found"
        res = []
        for match in fragment_pattern.finditer(res_text):
            res.append(
                {
                    "file": match.group(1),
                    "class": match.group(2),
                    "method": match.group(3),
                }
            )
            result_msg = "Found fragments"

        can_stop = res != []
        Path(output_dir, f"agent_fix_locations_{i}.json").write_text(
            json.dumps(res, indent=4)
        )
    return result_msg
