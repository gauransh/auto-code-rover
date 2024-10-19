from collections import defaultdict, namedtuple
from collections.abc import MutableMapping

from app.search import search_utils
from app.search.search_utils import SearchResult

LineRange = namedtuple("LineRange", ["start", "end"])

ClassIndexType = MutableMapping[str, list[tuple[str, LineRange]]]
ClassFuncIndexType = MutableMapping[
    str, MutableMapping[str, list[tuple[str, LineRange]]]
]
FuncIndexType = MutableMapping[str, list[tuple[str, LineRange]]]

RESULT_SHOW_LIMIT = 3


class SearchManager:
    def __init__(self, project_path: str):
        self.project_path = project_path
        
        # All attributes are initialized with their proper types
        self.parsed_files: List[str] = []  # ABSOLUTE paths of .py files
        self.class_index: ClassIndexType = {}
        self.class_func_index: ClassFuncIndexType = {}
        self.function_index: FuncIndexType = {}
        self.function_call_index: Dict[str, List[Tuple[str, LineRange]]] = defaultdict(list)
        self.class_bases: Dict[str, List[str]] = defaultdict(list)
        self.docstring_index: List[Tuple[str, str, str, LineRange]] = []
        
        # New attributes to address potential issues
        self.variable_index: Dict[str, List[Tuple[str, LineRange]]] = defaultdict(list)
        self.import_index: List[Tuple[str, List[str], str, LineRange]] = []
        self.decorator_index: List[Tuple[str, str, str, LineRange]] = []

        # Build the index after initializing all attributes
        self._build_index()

    def _build_index(self):
        """
        With all source code of the project, build two indexes:
            1. From class name to (source file, start line, end line)
            2. From function name to (source file, start line, end line)
        Since there can be two classes/functions with the same name, the mapping
        value is a list of tuples.
        This is for fast lookup whenever we receive a query.
        """
        self._update_indices(*self._build_python_index())

    def _update_indices(
        self,
        class_index: ClassIndexType,
        class_func_index: ClassFuncIndexType,
        function_index: FuncIndexType,
        parsed_files: list[str],
        variables: List[Tuple[str, str, LineRange]],
        imports: List[Tuple[str, str, LineRange]],
        decorators: List[Tuple[str, str, LineRange]],
        function_calls: List[Tuple[str, str, LineRange]],
        class_bases: Dict[str, List[str]],
        docstrings: List[Tuple[str, str, str, LineRange]],
    ) -> None:
        self.class_index.update(class_index)
        self.class_func_index.update(class_func_index)
        self.function_index.update(function_index)
        self.parsed_files.extend(parsed_files)
        self.docstring_index.extend(docstrings)
        self.import_index.extend(imports)
        self.decorator_index.extend(decorators)
        self.class_bases.update(class_bases)

        for var_name, file_path, line_range in variables:
            self.variable_index[var_name].append((file_path, line_range))

        for func_name, file_path, line_range in function_calls:
            self.function_call_index[func_name].append((file_path, line_range))

    def _build_python_index(
        self,
    ) -> tuple[ClassIndexType, ClassFuncIndexType, FuncIndexType, list[str]]:
        class_index: ClassIndexType = defaultdict(list)
        class_func_index: ClassFuncIndexType = defaultdict(lambda: defaultdict(list))
        function_index: FuncIndexType = defaultdict(list)

        py_files = search_utils.find_python_files(self.project_path)
        # holds the parsable subset of all py files
        parsed_py_files = []
        for py_file in py_files:
            file_info = search_utils.parse_python_file(py_file)
            if file_info is None:
                # parsing of this file failed
                continue
            parsed_py_files.append(py_file)
            # extract from file info, and form search index
            classes, class_to_funcs, top_level_funcs = file_info

            # (1) build class index
            for c, start, end in classes:
                class_index[c].append((py_file, LineRange(start, end)))

            # (2) build class-function index
            for c, class_funcs in class_to_funcs.items():
                for f, start, end in class_funcs:
                    class_func_index[c][f].append((py_file, LineRange(start, end)))

            # (3) build (top-level) function index
            for f, start, end in top_level_funcs:
                function_index[f].append((py_file, LineRange(start, end)))

        return class_index, class_func_index, function_index, parsed_py_files

    def file_line_to_class_and_func(
        self, file_path: str, line_no: int
    ) -> tuple[str | None, str | None]:
        """
        Given a file path and a line number, return the class and function name.
        If the line is not inside a class or function, return None.
        """
        # check whether this line is inside a class
        for class_name in self.class_func_index:
            func_dict = self.class_func_index[class_name]
            for func_name, func_info in func_dict.items():
                for file_name, (start, end) in func_info:
                    if file_name == file_path and start <= line_no <= end:
                        return class_name, func_name

        # not in any class; check whether this line is inside a top-level function
        for func_name in self.function_index:
            for file_name, (start, end) in self.function_index[func_name]:
                if file_name == file_path and start <= line_no <= end:
                    return None, func_name

        # this file-line is not recorded in any of the indexes
        return None, None

    def _search_func_in_class(
        self, function_name: str, class_name: str
    ) -> list[SearchResult]:
        """
        Search for the function name in the class.
        Args:
            function_name (str): Name of the function.
            class_name (str): Name of the class.
        Returns:
            The list of code snippets searched.
        """
        result: list[SearchResult] = []
        if class_name not in self.class_func_index:
            return result
        if function_name not in self.class_func_index[class_name]:
            return result
        for fname, (start, end) in self.class_func_index[class_name][function_name]:
            func_code = search_utils.get_code_snippets(fname, start, end, context=2)
            res = SearchResult(fname, class_name, function_name, func_code)
            result.append(res)
        return result

    def _search_func_in_all_classes(self, function_name: str) -> list[SearchResult]:
        """
        Search for the function name in all classes.
        Args:
            function_name (str): Name of the function.
        Returns:
            The list of code snippets searched.
        """
        result: list[SearchResult] = []
        for class_name in self.class_index:
            res = self._search_func_in_class(function_name, class_name)
            result.extend(res)
        return result

    def _search_top_level_func(self, function_name: str) -> list[SearchResult]:
        """
        Search for top-level function name in the entire project.
        Args:
            function_name (str): Name of the function.
        Returns:
            The list of code snippets searched.
        """
        result: list[SearchResult] = []
        if function_name not in self.function_index:
            return result

        for fname, (start, end) in self.function_index[function_name]:
            func_code = search_utils.get_code_snippets(fname, start, end, context=2)
            res = SearchResult(fname, None, function_name, func_code)
            result.append(res)
        return result

    def _search_func_in_code_base(self, function_name: str) -> list[SearchResult]:
        """
        Search for this function, from both top-level and all class definitions.
        """
        result: list[SearchResult] = []  # list of (file_name, func_code)
        # (1) search in top level
        top_level_res = self._search_top_level_func(function_name)
        class_res = self._search_func_in_all_classes(function_name)
        result.extend(top_level_res)
        result.extend(class_res)
        return result

    ###############################
    ### Interfaces ################
    ###############################

    # not search API - for writing patch
    # if we are searching for only a class when writing patch, likely we do not have enough info
    # the result can be too long, so we just show the first two
    def get_class_full_snippet(self, class_name: str) -> tuple[str, str, bool]:
        summary = f"Class {class_name} did not appear in the codebase."
        tool_result = f"Could not find class {class_name} in the codebase."

        if class_name not in self.class_index:
            return tool_result, summary, False
        # class name -> [(file_name, start_line, end_line)]
        search_res: list[SearchResult] = []
        for fname, (start, end) in self.class_index[class_name]:
            code = search_utils.get_code_snippets(fname, start, end, context=2)
            res = SearchResult(fname, class_name, None, code)
            search_res.append(res)

        if not search_res:
            return tool_result, summary, False

        # the good path
        # for all the searched result, append them and form the final result
        tool_result = f"Found {len(search_res)} classes with name {class_name} in the codebase:\n\n"
        summary = tool_result
        if len(search_res) > 2:
            tool_result += "Too many results, showing full code for 2 of them:\n"
        for idx, res in enumerate(search_res[:2]):
            res_str = res.to_tagged_str(self.project_path)
            tool_result += f"- Search result {idx + 1}:\n```\n{res_str}\n```"
        return tool_result, summary, True

    def search_class(self, class_name: str) -> tuple[str, str, bool]:
        # initialize them to error case
        summary = f"Class {class_name} did not appear in the codebase."
        tool_result = f"Could not find class {class_name} in the codebase."

        if class_name not in self.class_index:
            return tool_result, summary, False

        search_res: list[SearchResult] = []
        for fname, _ in self.class_index[class_name]:
            # there are some classes; we return their signatures
            code = search_utils.get_class_signature(fname, class_name)
            res = SearchResult(fname, class_name, None, code)
            search_res.append(res)

        if not search_res:
            # this should not happen, but just in case
            return tool_result, summary, False

        # the good path
        # for all the searched result, append them and form the final result
        tool_result = f"Found {len(search_res)} classes with name {class_name} in the codebase:\n\n"
        if len(search_res) > RESULT_SHOW_LIMIT:
            tool_result += "They appeared in the following files:\n"
            tool_result += SearchResult.collapse_to_file_level(
                search_res, self.project_path
            )
        else:
            for idx, res in enumerate(search_res):
                res_str = res.to_tagged_str(self.project_path)
                tool_result += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
        summary = f"The tool returned information about class `{class_name}`."
        return tool_result, summary, True

    def search_class_in_file(self, class_name, file_name: str) -> tuple[str, str, bool]:
        # (1) check whether we can get the file
        candidate_py_abs_paths = [f for f in self.parsed_files if f.endswith(file_name)]
        if not candidate_py_abs_paths:
            tool_output = f"Could not find file {file_name} in the codebase."
            summary = tool_output
            return tool_output, summary, False

        # (2) search for this class in the entire code base (we do filtering later)
        if class_name not in self.class_index:
            tool_output = f"Could not find class {class_name} in the codebase."
            summary = tool_output
            return tool_output, summary, False

        # (3) class is there, check whether it exists in the file specified.
        search_res: list[SearchResult] = []
        for fname, (start_line, end_line) in self.class_index[class_name]:
            if fname in candidate_py_abs_paths:
                class_code = search_utils.get_code_snippets(fname, start_line, end_line, context=2)
                res = SearchResult(fname, class_name, None, class_code)
                search_res.append(res)

        if not search_res:
            tool_output = f"Could not find class {class_name} in file {file_name}."
            summary = tool_output
            return tool_output, summary, False

        # good path; we have result, now just form a response
        tool_output = f"Found {len(search_res)} classes with name {class_name} in file {file_name}:\n\n"
        summary = tool_output
        for idx, res in enumerate(search_res):
            res_str = res.to_tagged_str(self.project_path)
            tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
        return tool_output, summary, True

    def search_method_in_file(
        self, method_name: str, file_name: str
    ) -> tuple[str, str, bool]:
        # (1) check whether we can get the file
        # supports both when file_name is relative to project root, and when
        # it is just a short name
        candidate_py_abs_paths = [f for f in self.parsed_files if f.endswith(file_name)]
        # print(candidate_py_files)
        if not candidate_py_abs_paths:
            tool_output = f"Could not find file {file_name} in the codebase."
            summary = tool_output
            return tool_output, summary, False

        # (2) search for this method in the entire code base (we do filtering later)
        search_res: list[SearchResult] = self._search_func_in_code_base(method_name)
        if not search_res:
            tool_output = f"The method {method_name} does not appear in the codebase."
            summary = tool_output
            return tool_output, summary, False

        # (3) filter the search result => they need to be in one of the files!
        filtered_res: list[SearchResult] = [
            res for res in search_res if res.file_path in candidate_py_abs_paths
        ]

        # (4) done with search, now prepare result
        if not filtered_res:
            tool_output = (
                f"There is no method with name `{method_name}` in file {file_name}."
            )
            summary = tool_output
            return tool_output, summary, False

        tool_output = f"Found {len(filtered_res)} methods with name `{method_name}` in file {file_name}:\n\n"
        summary = tool_output

        # when searching for a method in one file, it's rare that there are
        # many candidates, so we do not trim the result
        for idx, res in enumerate(filtered_res):
            res_str = res.to_tagged_str(self.project_path)
            tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
        return tool_output, summary, True

    def search_method_in_class(
        self, method_name: str, class_name: str
    ) -> tuple[str, str, bool]:
        if class_name not in self.class_index:
            tool_output = f"Could not find class {class_name} in the codebase."
            summary = tool_output
            return tool_output, summary, False

        # has this class, check its methods
        search_res: list[SearchResult] = self._search_func_in_class(
            method_name, class_name
        )
        if not search_res:
            tool_output = f"Could not find method {method_name} in class {class_name}`."
            summary = tool_output
            return tool_output, summary, False

        # found some methods, prepare the result
        tool_output = f"Found {len(search_res)} methods with name {method_name} in class {class_name}:\n\n"
        summary = tool_output

        # There can be multiple classes defined in multiple files, which contain the same method
        # still trim the result, just in case
        if len(search_res) > RESULT_SHOW_LIMIT:
            tool_output += f"Too many results, showing full code for {RESULT_SHOW_LIMIT} of them, and the rest just file names:\n"
        first_five = search_res[:RESULT_SHOW_LIMIT]
        for idx, res in enumerate(first_five):
            res_str = res.to_tagged_str(self.project_path)
            tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
        # for the rest, collect the file names into a set
        if rest := search_res[RESULT_SHOW_LIMIT:]:
            tool_output += "Other results are in these files:\n"
            tool_output += SearchResult.collapse_to_file_level(rest, self.project_path)
        return tool_output, summary, True

    def search_method(self, method_name: str) -> tuple[str, str, bool]:
        """
        Search for a method in the entire codebase.
        """
        search_res: list[SearchResult] = self._search_func_in_code_base(method_name)
        if not search_res:
            tool_output = f"Could not find method {method_name} in the codebase."
            summary = tool_output
            return tool_output, summary, False

        tool_output = f"Found {len(search_res)} methods with name {method_name} in the codebase:\n\n"
        summary = tool_output

        if len(search_res) > RESULT_SHOW_LIMIT:
            tool_output += "They appeared in the following files:\n"
            tool_output += SearchResult.collapse_to_file_level(
                search_res, self.project_path
            )
        else:
            for idx, res in enumerate(search_res):
                res_str = res.to_tagged_str(self.project_path)
                tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"

        return tool_output, summary, True

    def search_code(self, code_str: str) -> tuple[str, str, bool]:
        # attempt to search for this code string in all py files
        all_search_results: list[SearchResult] = []
        for file_path in self.parsed_files:
            searched_line_and_code: list[tuple[int, str]] = (
                search_utils.get_code_region_containing_code(file_path, code_str)
            )
            if not searched_line_and_code:
                continue
            for searched in searched_line_and_code:
                line_no, code_region = searched
                # from line_no, check which function and class we are in
                class_name, func_name = self.file_line_to_class_and_func(
                    file_path, line_no
                )
                res = SearchResult(file_path, class_name, func_name, code_region)
                all_search_results.append(res)

        if not all_search_results:
            tool_output = f"Could not find code {code_str} in the codebase."
            summary = tool_output
            return tool_output, summary, False

        # good path
        tool_output = f"Found {len(all_search_results)} snippets containing `{code_str}` in the codebase:\n\n"
        summary = tool_output

        if len(all_search_results) > RESULT_SHOW_LIMIT:
            tool_output += "They appeared in the following files:\n"
            tool_output += SearchResult.collapse_to_file_level(
                all_search_results, self.project_path
            )
        else:
            for idx, res in enumerate(all_search_results):
                res_str = res.to_tagged_str(self.project_path)
                tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
        return tool_output, summary, True

    def search_function_calls(self, function_name: str) -> List[SearchResult]:
        results = []
        if function_name not in self.function_call_index:
            return results
        for file_path, line_range in self.function_call_index[function_name]:
            code_snippet = search_utils.get_code_snippets(file_path, line_range.start, line_range.end, context=2)
            class_name, func_name = self.file_line_to_class_and_func(file_path, line_range.start)
            res = SearchResult(file_path, class_name, func_name, code_snippet)
            results.append(res)
        return results

    def search_import(self, module_name: str) -> List[SearchResult]:
        results = []
        for module, names, file_path, line_range in self.import_index:
            if module_name == module or module_name in names:
                code_snippet = search_utils.get_code_snippets(file_path, line_range.start, line_range.end, context=2)
                res = SearchResult(file_path, None, None, code_snippet)
                results.append(res)
        return results

    def search_variable(self, variable_name: str) -> List[SearchResult]:
        results = []
        if variable_name not in self.variable_index:
            return results
        for file_path, line_range in self.variable_index[variable_name]:
            code_snippet = search_utils.get_code_snippets(file_path, line_range.start, line_range.end, context=2)
            class_name, func_name = self.file_line_to_class_and_func(file_path, line_range.start)
            res = SearchResult(file_path, class_name, func_name, code_snippet)
            results.append(res)
        return results

    def search_code_in_file(
        self, code_str: str, file_name: str
    ) -> tuple[str, str, bool]:
        code_str = code_str.removesuffix(")")

        candidate_py_files = [f for f in self.parsed_files if f.endswith(file_name)]
        if not candidate_py_files:
            tool_output = f"Could not find file {file_name} in the codebase."
            summary = tool_output
            return tool_output, summary, False

        # start searching for code in the filtered files
        all_search_results: list[SearchResult] = []
        for file_path in candidate_py_files:
            searched_line_and_code: list[tuple[int, str]] = (
                search_utils.get_code_region_containing_code(file_path, code_str)
            )
            if not searched_line_and_code:
                continue
            for searched in searched_line_and_code:
                line_no, code_region = searched
                # from line_no, check which function and class we are in
                class_name, func_name = self.file_line_to_class_and_func(
                    file_path, line_no
                )
                res = SearchResult(file_path, class_name, func_name, code_region)
                all_search_results.append(res)

        if not all_search_results:
            tool_output = f"Could not find code {code_str} in file {file_name}."
            summary = tool_output
            return tool_output, summary, False

        # good path
        # There can be a lot of results, from multiple files.
        tool_output = f"Found {len(all_search_results)} snippets with code {code_str} in file {file_name}:\n\n"
        summary = tool_output
        if len(all_search_results) > RESULT_SHOW_LIMIT:
            tool_output += "They appeared in the following methods:\n"
            tool_output += SearchResult.collapse_to_method_level(
                all_search_results, self.project_path
            )
        else:
            for idx, res in enumerate(all_search_results):
                res_str = res.to_tagged_str(self.project_path)
                tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
        return tool_output, summary, True

    def retrieve_code_snippet(
        self, file_path: str, start_line: int, end_line: int
    ) -> str:
        return search_utils.get_code_snippets(file_path, start_line, end_line, context=2)

    def match_pattern(node: ast.AST, pattern: ast.AST) -> bool:
        if type(node) != type(pattern):
            return False
        for field, value in ast.iter_fields(pattern):
            node_value = getattr(node, field, None)
            if isinstance(value, ast.AST):
                if not match_pattern(node_value, value):
                    return False
            elif isinstance(value, list):
                if len(value) != len(node_value):
                    return False
                for v1, v2 in zip(value, node_value):
                    if not match_pattern(v1, v2):
                        return False
            else:
                if node_value != value:
                    return False
        return True

    def search_ast_pattern(self, pattern_code: str) -> List[SearchResult]:
        pattern_ast = ast.parse(pattern_code)
        results = []
        for file_path in self.parsed_files:
            try:
                file_content = pathlib.Path(file_path).read_text()
                tree = ast.parse(file_content)
            except Exception:
                continue
            for node in ast.walk(tree):
                if match_pattern(node, pattern_ast.body[0]):
                    start_lineno = node.lineno
                    end_lineno = node.end_lineno or node.lineno
                    code_snippet = search_utils.get_code_snippets(file_path, start_lineno, end_lineno, context=2)
                    class_name, func_name = self.file_line_to_class_and_func(file_path, start_lineno)
                    res = SearchResult(file_path, class_name, func_name, code_snippet)
                    results.append(res)
        return results

    def search_docstrings(self, search_text: str) -> List[SearchResult]:
        results = []
        for name, docstring, file_path, line_range in self.docstring_index:
            if search_text.lower() in docstring.lower():
                code_snippet = search_utils.get_code_snippets(file_path, line_range.start, line_range.end, context=2)
                class_name, func_name = self.file_line_to_class_and_func(file_path, line_range.start)
                res = SearchResult(file_path, class_name, func_name, code_snippet)
                results.append(res)
        return results

    def get_subclasses(self, class_name: str) -> List[str]:
        subclasses = []
        for subclass, bases in self.class_bases.items():
            if class_name in bases:
                subclasses.append(subclass)
        return subclasses