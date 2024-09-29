import json
import os
import shutil
from collections.abc import Callable, Mapping
from copy import deepcopy
from os.path import join as pjoin
from pathlib import Path

from docstring_parser import parse
from loguru import logger

from app import log
from app.analysis import sbfl
from app.analysis.sbfl import NoCoverageData
from app.api import agent_proxy, agent_write_locations, agent_write_patch
from app.data_structures import FunctionCallIntent, MessageThread
from app.log import log_exception, print_acr
from app.search.search_manage import SearchManager

# from app.api.python.validation import PythonValidator
from app.task import Task
from string import Template
from app.model import common

GEN_ONE_NO_REF_SIMPLE= Template("""
You are provided with the following Python function stub and docstring. 
You want to ensure that when the function is implemented, it complies with the specification given in the docstring:
${codeStubAndDocstring}

Your task is to write a ${toGenerateFull}. The ${toGenerateShort} should be in Python and consist of exactly five assert statement. 
A Python comment explaining the ${toGenerateShort}'s meaning should precede it.

For variables, the ${toGenerateShort} should only use the input parameters defined in the function stub and a hypothetical return value of the function, 
which we'll assume is stored in a variable `return_value`. For string manipulation, Python's re (regular expressions) library can be used. 
If other Python standard library functions are necessary, include the corresponding imports. However, refrain from using external libraries or 
calling the function itself within the ${toGenerateShort}.

If the ${toGenerateShort} calls any functions, they should only be those from the functional subset of Python. By this, we mean functions that are pure 
(i.e., no side effects, depends only on input values) such as `all()`, `len()`, `map()`, `filter()`, etc.

Although the ${toGenerateShort} should be less computationally complex than the function itself and relatively simple, it should not be trivial. 
It should encapsulate an aspect of the function output specification without implementing the function itself and should be easily readable by a human.

While not trivial, your ${toGenerateShort} should still be very simple and short. It should be a single line of code that is not too long, and it should capture 
only one aspect of the function's behavior, not all of it. 

For example, if the goal of the function were to sort a list, you might write a ${toGenerateShort} that checks that the elements in the list are in sorted order, 
or you might write a ${toGenerateShort} that checks that the list is the same length as the input list. You would not write a ${toGenerateShort} that checks both of these things.

The format of your response should be:
# Comment explaining what aspect of the function the ${toGenerateFull} checks
CODE FOR EXACTLY FIVE ${toGenerateShortCaps} USING ASSERT GOES HERE

The ${toGenerateShort} should hold true whenever the function executes successfully as specified in the docstring, regardless of the eventual internal implementation of the function.
""")

POST_COND_TO_USE = Template("For variables, only use the function inputs and the return value of the function. You can use python's re (regular expressions) if needed to deal with strings. Do not call ${entrypoint} itself in the postcondition. Instead, assume that the function has already been called and its return value is available in a variable called `return_value` that you can use. In the postcondition, only use functions that are part of the functional subset of python (e.g., all(), len(), map(), filter(), etc.)")


class PostconditionGenerator:
    def __init__(self, description, file_context=None, signature=None, implementation=None, comments=None, pull_request_info=None):
        self.description = description
        self.file_context = file_context
        self.signature = signature
        self.implementation = implementation
        self.comments = comments
        self.pull_request_info = pull_request_info

    def generate_postcondition(self):
        
        context = {
            'codeStubAndDocstring': self.description,
            'toGenerateFull': 'postcondition',
            'toGenerateShort': 'postcondition',
            'toGenerateShortCaps': 'POSTCONDITION',
            'toGenerateGoal': 'does',
            'promptAdds': '',
        }

        if self.pull_request_info:
            context['promptAdds'] = f"""
            Pull Request Body: {self.pull_request_info['pr_body']}
            Pull Request Code Diff: {self.pull_request_info['diff']}
            Pull Request Commits: {self.pull_request_info['commits']}
            Pull Request Additions: {self.pull_request_info['additions']}
            Pull Request Deletions: {self.pull_request_info['deletions']}
            Pull Request Changed Files: {self.pull_request_info['changed_files']}
            """
            context['codeStubAndDocstring'] += context['promptAdds']

        if self.file_context:
            context['codeStubAndDocstring'] += f"\nFile context: {self.file_context}"
        if self.signature:
            context['codeStubAndDocstring'] += f"\n{self.function_name} signature: {self.signature}"
        if self.implementation:
            context['codeStubAndDocstring'] += f"\nFunction implementation: {self.implementation}"
        if self.comments:
            context['codeStubAndDocstring'] += f"\nComments from users: {self.comments}"

        prompt = GEN_ONE_NO_REF_SIMPLE.safe_substitute(context)
        print_acr(f"Constructed prompt for postcondition generation: {prompt}")

        SYSTEM_PROMPT = """You are a helpful assistant that generates **symbolic** postcondition a text into json format."""

        msg_thread = MessageThread()
        msg_thread.add_system(SYSTEM_PROMPT)
        msg_thread.add_user(prompt)
        res_text, *_ = common.SELECTED_MODEL.call(msg_thread.to_msg(), response_format="json_object")
        if res_text:
            output = res_text.strip().split('\n') # Split the string into a list of assertions
            print_acr(f"Successfully generated postcondition from API\n {output}")
            return output
        else:
            print_acr("API returned no content for postcondition generation.")
        return ["No valid response from API."]


class ProjectApiManager:
    ################# State machine specific ################
    # NOTE: this section is for state machine; APIs in stratified mode are specified
    # in agent_api_selector.py
    api_functions = [
        "search_class",
        "search_class_in_file",
        "search_method",
        "search_method_in_class",
        "search_method_in_file",
        "search_code",
        "search_code_in_file",
        "write_patch",
    ]

    
    def generate_postcondition_(self, description=None, implementation=None, file_context=None, signature=None, external_functions_info=None, comments=None, pull_request_info=None, only_pr=False):
        generator = PostconditionGenerator(description, file_context, signature, implementation, comments, pull_request_info)
        return generator.generate_postcondition()

    def generate_postcondition(self,description=None, implementation=None, file_context=None, signature=None, external_functions_info=None, comments=None, pull_request_info=None, only_pr=False):
        
        context = {
            'codeStubAndDocstring': description or '',
            'toGenerateFull': 'postcondition',
            'toGenerateShort': 'postcondition',
            'toGenerateShortCaps': 'POSTCONDITION',
            'toGenerateGoal': 'does',
            'promptAdds': '',
        }

        if only_pr and pull_request_info:
            context['codeStubAndDocstring'] += f"""
                            Pull Request Body: {pull_request_info['pr_body']}
                            Pull Request Code Diff: {pull_request_info['diff']}
                            Pull Request Commits: {pull_request_info['commits']}
                            Pull Request Additions: {pull_request_info['additions']}
                            Pull Request Deletions: {pull_request_info['deletions']}
                            Pull Request Changed Files: {pull_request_info['changed_files']}
                            """
            prompt = GEN_ONE_NO_REF_SIMPLE.safe_substitute(context)
            print_acr(f"Prompt looks like: {prompt}")
        else:
            if pull_request_info:
                context['promptAdds'] = f"""
                Pull Request Body: {pull_request_info['pr_body']}
                Pull Request Code Diff: {pull_request_info['diff']}
                Pull Request Commits: {pull_request_info['commits']}
                Pull Request Additions: {pull_request_info['additions']}
                Pull Request Deletions: {pull_request_info['deletions']}
                Pull Request Changed Files: {pull_request_info['changed_files']}
                """
                context['codeStubAndDocstring'] = str(context['codeStubAndDocstring']) + context['promptAdds']

            if file_context:
                context['codeStubAndDocstring'] += f"\nFile context: {file_context}"
            if signature:
                context['codeStubAndDocstring'] += f"\n{function_name} signature: {signature}"
            if implementation:
                context['codeStubAndDocstring'] += f"\nFunction implementation: {implementation}"
            if comments:
                context['codeStubAndDocstring'] += f"\nComments from users: {comments}"

            prompt = GEN_ONE_NO_REF_SIMPLE.safe_substitute(context)
            print_acr(f"Constructed prompt for postcondition generation: {prompt}")
        SYSTEM_PROMPT = """You are a helpful assistant that generates **symbolic** postcondition a text into json format."""

        msg_thread = MessageThread()
        msg_thread.add_system(SYSTEM_PROMPT)
        msg_thread.add_user(prompt)
        res_text, *_ = common.SELECTED_MODEL.call(msg_thread.to_msg(), response_format="json_object")
        if res_text:
            output = res_text.strip().split('\n') # Split the string into a list of assertions
            print_acr(f"Successfully generated postcondition from API\n {output}")
            return output
        else:
            print_acr("API returned no content for postcondition generation.")
        return ["No valid response from API."]

    
    def _generate_postcondition(self, description):
        prompt = f"""
        Given the following natural language description of a method:
        "{description}".
        Write a **symbolic** postcondition for the function consisting of exactly one assert statement and add three more assert statements.
        """
        
        SYSTEM_PROMPT = """You are a helpful assistant that generates **symbolic** postcondition a text into json format."""

        msg_thread = MessageThread()
        msg_thread.add_system(SYSTEM_PROMPT)
        msg_thread.add_user(prompt)
        res_text, *_ = common.SELECTED_MODEL.call(msg_thread.to_msg(), response_format="json_object")
        if res_text:
            output = res_text.strip().split('\n') # Split the string into a list of assertions
            print_acr(f"Successfully generated postcondition from API\n {output}")
            return output
        else:
            print_acr("API returned no content for postcondition generation.")
        return ["No valid response from API."]
    

    def generate_postconditions(self) -> tuple[str, str, bool]:
        """Generate postconditions for the current task."""
        try:
            postconditions = self.generate_postcondition_(self.task.get_issue_statement())

            postcondition_str = "\n".join(postconditions)
            tool_output = f"Generated postconditions:\n{postcondition_str}"
            summary = "Successfully generated postconditions."
            return summary,tool_output, True
        except Exception as e:
            tool_output = f"Error: {str(e)}"
            summary = f"Failed to generate postconditions for {str(e)}."
            return tool_output, summary, False


    def next_tools(self) -> list[str]:
        """
        Return the list of tools that should be used in the next round.
        """
        search_tools = [
            "search_class",
            "search_class_in_file",
            "search_method",
            "search_method_in_class",
            "search_method_in_file",
            "search_code",
            "search_code_in_file",
        ]
        all_tools = search_tools + ["write_patch"]
        if not self.curr_tool:
            # this means we are at the beginning of the conversation
            # you have to start from doing some search
            return search_tools

        state_machine = {
            "search_class": search_tools,
            "search_class_in_file": search_tools,
            "search_method": all_tools,
            "search_method_in_class": all_tools,
            "search_method_in_file": all_tools,
            "search_code": all_tools,
            "search_code_in_file": all_tools,
            "write_patch": [],
        }
        return state_machine[self.curr_tool]

    def __init__(self, task: Task, output_dir: str):
        # for logging of this task instance
        self.task = task

        # where to write our output
        self.output_dir = os.path.abspath(output_dir)

        self.task.setup_project()
        # self.setup_project(self.task)

        # build search manager
        self.search_manager = SearchManager(self.task.project_path)

        # keeps track which tools is currently being used
        self.curr_tool: str | None = None

        # record the sequence of tools used, and their return status
        self.tool_call_sequence: list[Mapping] = []

        # record layered API calls
        self.tool_call_layers: list[list[Mapping]] = []

        # record cost and token information
        self.cost: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    @classmethod
    def get_short_func_summary_for_openai(cls) -> str:
        """
        Get a short summary of all tool functions.
        Intended to be used for constructing the initial system prompt.
        """
        summary = ""
        for fname in cls.api_functions:
            if not hasattr(cls, fname):
                continue
            func_obj = getattr(cls, fname)
            doc = parse(func_obj.__doc__)
            short_desc = (
                doc.short_description if doc.short_description is not None else ""
            )
            summary += f"\n- {fname}: {short_desc}"
        return summary

    @classmethod
    def get_full_funcs_for_openai(cls, tool_list: list[str]) -> list[dict]:
        """
        Return a list of function objects which can be sent to OpenAI for
        the function calling feature.

        Args:
            tool_list (List[str]): The available tools to generate doc for.
        """
        tool_template = {
            "type": "function",
            "function": {
                "name": "",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {},  # mapping from para name to type+description
                    "required": [],  # name of required parameters
                },
            },
        }
        all_tool_objs = []

        for fname in tool_list:
            if not hasattr(cls, fname):
                continue
            tool_obj = deepcopy(tool_template)
            tool_obj["function"]["name"] = fname
            func_obj = getattr(cls, fname)
            # UPDATE: we only parse docstring now
            # there are two sources of information:
            # 1. the docstring
            # 2. the function signature
            # Docstring is where we get most of the textual descriptions; for accurate
            # info about parameters (whether optional), we check signature.

            ## parse docstring
            doc = parse(func_obj.__doc__)
            short_desc = (
                doc.short_description if doc.short_description is not None else ""
            )
            long_desc = doc.long_description if doc.long_description is not None else ""
            description = short_desc + "\n" + long_desc
            tool_obj["function"]["description"] = description
            doc_params = doc.params
            for doc_param in doc_params:
                param_name = doc_param.arg_name
                if param_name == "self":
                    continue
                typ = doc_param.type_name
                desc = doc_param.description
                is_optional = doc_param.is_optional
                # now add this param to the tool object
                tool_obj["function"]["parameters"]["properties"][param_name] = {
                    "type": typ,
                    "description": desc,
                }
                if not is_optional:
                    tool_obj["function"]["parameters"]["required"].append(param_name)

            all_tool_objs.append(tool_obj)

        return all_tool_objs

    def dispatch_intent(
        self,
        intent: FunctionCallIntent,
        message_thread: MessageThread,
        print_callback: Callable[[dict], None] | None = None,
    ) -> tuple[str, str, bool]:
        """Dispatch a function call intent to actually perform its action.

        Args:
            intent (FunctionCallIntent): The intent to dispatch.
            message_thread (MessageThread): the current message thread,
                since some tools require it.
        Returns:
            The result of the action.
            Also a summary that should be communicated to the model.
        """
        if (intent.func_name not in self.api_functions) and (
            intent.func_name not in ["get_class_full_snippet", "propose_locs"]
        ):
            error = f"Unknown function name {intent.func_name}."
            summary = "You called a tool that does not exist. Please only use the tools provided."
            return error, summary, False
        func_obj = getattr(self, intent.func_name)
        try:
            # ready to call a function
            self.curr_tool = intent.func_name
            if intent.func_name in ["write_patch", "propose_locs"]:
                # these two functions require the message thread
                call_res = func_obj(message_thread, print_callback=print_callback)
            else:
                call_res = func_obj(**intent.arg_values)
        except Exception as e:
            # TypeError can happen when the function is called with wrong parameters
            # we just return the error message as the call result
            log_exception(e)
            error = str(e)
            summary = "The tool returned error message."
            call_res = (error, summary, False)

        logger.debug("Result of dispatch_intent: {}", call_res)

        # record this call and its result separately
        _, _, call_is_ok = call_res
        self.tool_call_sequence.append(intent.to_dict_with_result(call_is_ok))

        if not self.tool_call_layers:
            self.tool_call_layers.append([])
        self.tool_call_layers[-1].append(intent.to_dict_with_result(call_is_ok))

        return call_res

    def start_new_tool_call_layer(self):
        self.tool_call_layers.append([])

    def dump_tool_call_sequence_to_file(self):
        """Dump the sequence of tool calls to a file."""
        tool_call_file = pjoin(self.output_dir, "tool_call_sequence.json")
        with open(tool_call_file, "w") as f:
            json.dump(self.tool_call_sequence, f, indent=4)

    def dump_tool_call_layers_to_file(self):
        """Dump the layers of tool calls to a file."""
        tool_call_file = pjoin(self.output_dir, "tool_call_layers.json")
        with open(tool_call_file, "w") as f:
            json.dump(self.tool_call_layers, f, indent=4)

    ###################################################################
    ########################## API functions ##########################
    ###################################################################

    def fault_localization(self) -> tuple[str, str, bool]:
        """Localize the faulty code snippets by executing test cases.

        Perform fault localization by running the passing and failing test-cases.
        Returns a list of code snippets that are likely to be related to the issue.
        """
        sbfl_result_file = Path(self.output_dir, "sbfl_result.json")
        sbfl_method_result_file = Path(self.output_dir, "sbfl_result_method.json")

        log_file = None
        try:
            test_file_names, ranked_lines, log_file = sbfl.run(self.task)
        except NoCoverageData as e:
            sbfl_result_file.write_text("")
            sbfl_method_result_file.write_text("")

            log_file = e.testing_log_file

            tool_output = "Error in running localization tool"
            summary = tool_output
            return tool_output, summary, False
        finally:
            if log_file is not None:
                shutil.move(log_file, pjoin(self.output_dir, "run_developer_tests.log"))

        ranked_ranges_abs = sbfl.collate_results(ranked_lines, test_file_names)
        ranked_methods_abs = sbfl.map_collated_results_to_methods(ranked_ranges_abs)

        def relativize_filename(tup: tuple) -> tuple:
            file = tup[0]
            relative_file = os.path.relpath(file, self.task.project_path)
            return (relative_file,) + tup[1:]

        ranked_ranges = [relativize_filename(t) for t in ranked_ranges_abs]
        ranked_methods = [relativize_filename(t) for t in ranked_methods_abs]

        sbfl_result_file.write_text(json.dumps(ranked_ranges, indent=4))

        sbfl_method_result_file.write_text(json.dumps(ranked_methods, indent=4))

        log.log_and_print(f"SBFL result (lines): {ranked_ranges}")
        log.log_and_print(f"SBFL result (methods): {ranked_methods}")

        return self._form_sbfl_output(ranked_methods)

    @classmethod
    def _form_sbfl_output(cls, ranked_methods) -> tuple[str, str, bool]:
        if not ranked_methods:
            # empty sbfl results
            tool_output = "Localization could not produce any output."
            summary = tool_output
            return tool_output, summary, False

        if len(ranked_methods) > 5:
            ranked_methods = ranked_methods[:5]

        tool_output = f"Top-{len(ranked_methods)} suspicious methods:\n"
        for idx, (file, class_name, method_name, _) in enumerate(ranked_methods):

            res_str = f"<file>{file}</file>"
            if class_name:
                res_str += f" <class>{class_name}</class>"
            if method_name:
                res_str += f" <func>{method_name}</func>"

            tool_output += f"Suspicious method #{idx + 1}:\n{res_str}\n\n"

        summary = f"Returned top-{len(ranked_methods)} suspicious methods."

        return tool_output, summary, True

    # not a search API - just to get full class definition when bug_location only specifies a class
    def get_class_full_snippet(self, class_name: str):
        return self.search_manager.get_class_full_snippet(class_name)

    def search_class(self, class_name: str) -> tuple[str, str, bool]:
        """Search for a class in the codebase.

        Only the signature of the class is returned. The class signature
        includes class name, base classes, and signatures for all of its methods/properties.

        Args:
            class_name (string): Name of the class to search for.

        Returns:
            string: the class signature in string if success;
                    an error message if the class cannot be found.
            string: a message summarizing the method.
        """
        return self.search_manager.search_class(class_name)

    def search_class_in_file(
        self, class_name: str, file_name: str
    ) -> tuple[str, str, bool]:
        """Search for a class in a given file.

        Returns the actual code of the entire class definition.

        Args:
            class_name (string): Name of the class to search for.
            file_name (string): The file to search in. Must be a valid python file name.

        Returns:
            part 1 - the searched class code or error message.
            part 2 - summary of the tool call.
        """
        return self.search_manager.search_class_in_file(class_name, file_name)

    def search_method_in_file(
        self, method_name: str, file_name: str
    ) -> tuple[str, str, bool]:
        """Search for a method in a given file.

        Returns the actual code of the method.

        Args:
            method_name (string): Name of the method to search for.
            file_name (string): The file to search in. Must be a valid python file name.

        Returns:
            part 1 - the searched code or error message.
            part 2 - summary of the tool call.
        """
        return self.search_manager.search_method_in_file(method_name, file_name)

    def search_method_in_class(
        self, method_name: str, class_name: str
    ) -> tuple[str, str, bool]:
        """Search for a method in a given class.

        Returns the actual code of the method.

        Args:
            method_name (string): Name of the method to search for.
            class_name (string): Consider only methods in this class.

        Returns:
            part 1 - the searched code or error message.
            part 2 - summary of the tool call.
        """
        return self.search_manager.search_method_in_class(method_name, class_name)

    def search_method(self, method_name: str) -> tuple[str, str, bool]:
        """Search for a method in the entire codebase.

        Returns the actual code of the method.

        Args:
            method_name (string): Name of the method to search for.

        Returns:
            part 1 - the searched code or error message.
            part 2 - summary of the tool call.
        """
        return self.search_manager.search_method(method_name)

    def search_code(self, code_str: str) -> tuple[str, str, bool]:
        """Search for a code snippet in the entire codebase.

        Returns the method that contains the code snippet, if it is found inside a file.
        Otherwise, returns the region of code surrounding it.

        Args:
            code_str (string): The code snippet to search for.

        Returns:
            The region of code containing the searched code string.
        """
        return self.search_manager.search_code(code_str)

    def search_code_in_file(
        self, code_str: str, file_name: str
    ) -> tuple[str, str, bool]:
        """Search for a code snippet in a given file file.

        Returns the entire method that contains the code snippet.

        Args:
            code_str (string): The code snippet to search for.
            file_name (string): The file to search in. Must be a valid python file name in the project.

        Returns:
            The method code that contains the searched code string.
        """
        return self.search_manager.search_code_in_file(code_str, file_name)

    def write_patch(
        self,
        message_thread: MessageThread,
        print_callback: Callable[[dict], None] | None = None,
    ) -> tuple[str, str, bool]:
        """Based on the current context, ask another agent to write a patch.

        When you think the current information is sufficient to write a patch, invoke this tool.

        The tool returns a patch based on the current available information.
        """
        tool_output = agent_write_patch.run_with_retries(
            message_thread,
            self.output_dir,
            self.task,
            # self.validator,
            print_callback=print_callback,
        )
        summary = "The tool returned the patch written by another agent."
        # The return status of write_patch does not really matter, so we just use True here
        return tool_output, summary, True

    def propose_locs(
        self,
        message_thread: MessageThread,
        print_callback: Callable[[dict], None] | None = None,
    ) -> tuple[str, str, bool]:
        """Propose locations for fixing the bug.

        Based on the current context, ask another agent to propose locations for fixing the bug.

        The tool returns a list of locations that are likely to be related to the issue.
        """
        tool_output = agent_write_locations.run_with_retries(
            message_thread,
            self.output_dir,
            self.task,
            # self.validator,
            print_callback=print_callback,
        )
        summary = "The tool returned the locations proposed by another agent."
        # The return status of propose_locs does not really matter, so we just use True here
        return tool_output, summary, True

    def proxy_apis(self, text: str) -> tuple[str | None, str, list[MessageThread]]:
        """Proxy APIs to another agent."""
        tool_output, new_thread = agent_proxy.run_with_retries(
            text
        )  # FIXME: type of `text`
        if tool_output is None:
            summary = "The tool returned nothing. The main agent probably did not provide enough clues."
        else:
            summary = "The tool returned the selected search APIs in json format generaetd by another agent."
        return tool_output, summary, new_thread
