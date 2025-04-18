import re
from queue import Queue
from typing import List, Tuple, Optional

import autopep8

from MVP.refactored.backend.box_functions.box_function import BoxFunction
from MVP.refactored.backend.code_generation.renamer import Renamer
from MVP.refactored.backend.hypergraph.hypergraph import Hypergraph
from MVP.refactored.backend.hypergraph.hypergraph_manager import HypergraphManager
from MVP.refactored.backend.hypergraph.node import Node
from MVP.refactored.frontend.components.custom_canvas import CustomCanvas
from MVP.refactored.backend.box_functions.box_function import functions as predefined_functions


class CodeGenerator:
    @classmethod
    def generate_code(cls, canvas: CustomCanvas, canvasses: dict[str, CustomCanvas], main_diagram) -> str:
        code_parts: dict[BoxFunction, list[int]] = cls.get_all_code_parts(canvas, canvasses, main_diagram)
        file_content = cls.get_imports([f.code for f in code_parts.keys()]) + "\n\n"

        box_functions: dict[BoxFunction, set[str]] = {}

        for box_function in code_parts.keys():
            renamer = Renamer()
            variables = set()
            variables.update(renamer.find_globals(box_function.code))
            variables.update(renamer.find_function_names(box_function.code))
            box_functions[box_function] = variables

        function_list, renamed_functions = cls.rename(box_functions)
        function_list = cls.remove_meta(function_list)
        function_list = cls.remove_imports(function_list)

        file_content += "\n".join(function_list)

        file_content += "\n" + cls.construct_main_function(canvas, renamed_functions)

        with open("diagram.py", "w") as file:
            file.write(autopep8.fix_code(file_content))

        return autopep8.fix_code(file_content)

    @classmethod
    def get_all_code_parts(cls, canvas: CustomCanvas, canvasses: dict[str, CustomCanvas], main_diagram) -> \
            dict[BoxFunction, list[int]]:
        code_parts: dict[BoxFunction, list[int]] = dict()
        for box in canvas.boxes:
            if str(box.id) in canvasses:
                code_parts.update(cls.get_all_code_parts(canvasses.get(str(box.id)), canvasses, main_diagram))
            else:
                code_template = main_diagram.label_content[box.label_text]
                substitution_dict = {}
                print(box.label_text)
                print("#############################")
                if "layer" in box.label_text or "transformer":
                    substitution_dict = {'neurons': box.neurons, 'activation': box.activation, 'inputs': box.inputs,
                                         'optimizer': box.optimizer, 'loss': box.loss, 'metrics': box.metrics,
                                         'outputs': box.outputs,
                                         'num_heads': box.num_heads, 'num_encoder_layers': box.num_encoder_layers,
                                         'num_decoder_layers': box.num_decoder_layers}
                box_function = BoxFunction(box.label_text, code=code_template, substitution_dict=substitution_dict)
                box.box_function = box_function
                if box_function in code_parts:
                    code_parts[box_function].append(box.id)
                else:
                    code_parts[box_function] = [box.id]
        return code_parts

    @classmethod
    def get_imports(cls, code_parts: list[str]) -> str:
        regex = r"(^import .+)|(^from .+)"
        imports = set()
        for part in code_parts:
            code_imports = re.finditer(regex, part, re.MULTILINE)
            for code_import in code_imports:
                imports.add(code_import.group())
        return "\n".join(imports)

    @classmethod
    def rename(cls, names: dict[BoxFunction, set[str]]) -> tuple[list[str], dict[BoxFunction, str]]:
        renamed_code_parts: list[str] = list()
        renamed_functions: dict[BoxFunction, str] = dict()
        for i, (box_function, names) in enumerate(names.items()):
            renamer = Renamer()
            code_part = box_function.code
            for name in names:
                if name == "meta":
                    continue
                new_name = f'{name}_{i}'
                if name == "invoke":
                    renamed_functions[box_function] = new_name
                code_part = renamer.refactor_code(code_part, name, new_name)
            renamed_code_parts.append(code_part)
        return renamed_code_parts, renamed_functions

    @classmethod
    def remove_imports(cls, code_parts: list[str]) -> list[str]:
        regex = r"(^import .+)|(^from .+)"
        regex2 = r"^\n+"
        code_parts_without_imports = []

        for part in code_parts:
            cleaned_part = re.sub(regex, "", part, flags=re.MULTILINE)
            cleaned_part = re.sub(regex2, "", cleaned_part)
            code_parts_without_imports.append(cleaned_part)
        return code_parts_without_imports

    @classmethod
    def remove_meta(cls, code_parts: list[str]) -> list[str]:
        regex = r"^meta\s=\s{[\s\S]+?}"
        regex2 = r"^\n+"
        code_parts_without_meta = []

        for part in code_parts:
            cleaned_part = re.sub(regex, "", part, flags=re.MULTILINE)
            cleaned_part = re.sub(regex2, "", cleaned_part)
            code_parts_without_meta.append(cleaned_part)
        return code_parts_without_meta

    @classmethod
    def construct_main_function(cls, canvas: CustomCanvas, renamed_functions: dict[BoxFunction, str]) -> str:
        # main_function = ""
        # hypergraph: Hypergraph = HypergraphManager().get_graph_by_id(canvas.id)
        # input_nodes: set[Node] = set(hypergraph.get_node_by_input(input_id) for input_id in hypergraph.inputs)
        # # input_nodes = sorted(input_nodes, key=lambda input_node: canvas.get_box_by_id(input_node.id).y)                # original sort
        # input_nodes = sorted(input_nodes, key=lambda input_node: CodeGenerator.get_item_y(canvas, input_node.id))
        # main_function += cls.create_definition_of_main_function(input_nodes)
        # nodes_queue: Queue[Node] = Queue()
        # node_input_count_check: dict[int, int] = dict()
        #
        # for node in input_nodes:
        #     node_input_count_check[node.id] = 0
        #     for node_input in node.inputs:
        #         if hypergraph.get_node_by_output(node_input) is None:
        #             node_input_count_check[node.id] += 1
        #     if node_input_count_check[node.id] == len(node.inputs):
        #         nodes_queue.put(node)
        #
        # while len(input_nodes) > 0:
        #     input_nodes = cls.get_children_nodes(input_nodes,
        #                                          node_input_count_check)
        #     # input_nodes = sorted(input_nodes, key=lambda input_node: canvas.get_box_by_id(input_node.id).y)
        #     input_nodes = sorted(input_nodes, key=lambda input_node: CodeGenerator.get_item_y(canvas, input_node.id))
        #     for node in input_nodes:
        #         nodes_queue.put(node)
        #
        # main_function_content, function_result_variables = cls.create_main_function_content(
        #                                                             canvas, nodes_queue, renamed_functions, hypergraph)
        # main_function_return = cls.create_main_function_return(function_result_variables, hypergraph)
        #
        # main_function += main_function_content
        # main_function += main_function_return
        # return main_function
        main_function = ""
        hypergraph: Hypergraph = HypergraphManager().get_graph_by_id(canvas.id)

        # Get the input nodes (those corresponding to hypergraph.inputs)
        input_nodes: set = set(hypergraph.get_node_by_input(input_id) for input_id in hypergraph.inputs)
        input_nodes = sorted(input_nodes, key=lambda input_node: CodeGenerator.get_item_y(canvas, input_node.id))
        main_function += cls.create_definition_of_main_function(input_nodes)

        # Instead of building a Queue manually, perform a topological sort on all nodes.
        sorted_nodes = cls.topological_sort(hypergraph.nodes, hypergraph)

        main_function_content, function_result_variables = cls.create_main_function_content_from_sorted_list(
            canvas, sorted_nodes, renamed_functions, hypergraph
        )
        main_function_return = cls.create_main_function_return(function_result_variables, hypergraph)

        # main_function += main_function_content
        # main_function += main_function_return

        lines = main_function_content.splitlines()
        if lines:
            # Leave the first line unmodified.
            indented_body = lines[0] + "\n" + "\n".join(
                "\t" + line if line.strip() != "" else "" for line in lines[1:])
        else:
            indented_body = ""

        indented_return = "\n".join(
            "\t" + line if line.strip() != "" else "" for line in main_function_return.splitlines())

        main_function += indented_body + "\n" + indented_return + "\n"
        return main_function

    def topological_sort(nodes: List, hypergraph) -> List:
        # Create a mapping from node id to node.
        node_by_id = {node.id: node for node in nodes}
        # Initialize in-degrees for each node.
        in_degree = {node.id: 0 for node in nodes}
        for node in nodes:
            for inp in node.inputs:
                parent = hypergraph.get_node_by_output(inp)
                if parent is not None:
                    in_degree[node.id] += 1
        # Start with nodes having zero in-degree.
        sorted_nodes = []
        queue = [node_by_id[nid] for nid, deg in in_degree.items() if deg == 0]
        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            for child in hypergraph.get_node_children_by_node(node):
                in_degree[child.id] -= 1
                if in_degree[child.id] == 0:
                    queue.append(child)
        return sorted_nodes

    # @classmethod
    # def create_main_function_content_from_sorted_list(cls, canvas: CustomCanvas, sorted_nodes: List,
    #                                                   renamed_functions: dict[BoxFunction, str],
    #                                                   hypergraph: Hypergraph) -> Tuple[str, dict]:
    #     function_result_variables: dict[int, str] = dict()
    #     input_index = 1
    #     result_index = 1
    #     content = ""
    #
    #     # for nodes with multiple outputs track the next available index
    #     function_output_index: dict[int, int] = {node.id: 0 for node in sorted_nodes if len(node.outputs) > 1}
    #
    #     for node in sorted_nodes:
    #
    #         # try to get the box function for this node if is None handle as a spider
    #         current_box_function = canvas.get_box_function(node.id)
    #         if current_box_function is None:
    #
    #             box = canvas.get_box_by_id(node.id)
    #             if box is not None and box.sub_diagram is not None:
    #                 print("sub diagram detected for node id:", node.id)
    #
    #                 sub_hypergraph: Hypergraph = HypergraphManager().get_graph_by_id(box.sub_diagram.id)
    #                 sub_sorted_nodes = CodeGenerator.topological_sort(sub_hypergraph.nodes, sub_hypergraph)
    #
    #                 if node.inputs and node.inputs[0] in function_result_variables:
    #                     container_input = function_result_variables[node.inputs[0]]
    #                 else:
    #                     container_input = f"res_{result_index - 1}"
    #
    #                 # Recursively generate code for the sub-diagram.
    #                 sub_code, sub_function_vars = cls.create_main_function_content_from_sorted_list(
    #                     box.sub_diagram, sub_sorted_nodes, renamed_functions, sub_hypergraph)
    #
    #                 sub_code = sub_code.replace("input_1", container_input, 1)
    #
    #                 indented_sub_code = "\n".join("\t" + line for line in sub_code.splitlines())
    #                 content += "\n" + indented_sub_code
    #
    #                 if sub_sorted_nodes:
    #                     container_output = sub_function_vars[sub_sorted_nodes[-1].id]
    #                 else:
    #                     container_output = "None"
    #                 function_result_variables[node.id] = container_output
    #                 continue
    #
    #             if any(spider.id == node.id for spider in canvas.spiders):
    #                 if len(node.inputs) == 1:
    #                     spider_var = f"res_{result_index}"
    #                     result_index += 1
    #                     if node.inputs:
    #                         parent_var = function_result_variables.get(node.inputs[0], f"input_{input_index}")
    #                         if node.inputs[0] not in function_result_variables:
    #                             input_index += 1
    #                     else:
    #                         parent_var = f"input_{input_index}"
    #                         input_index += 1
    #                     line = f"{spider_var} = {parent_var}\n\t"
    #                     content += f"# Spider node {node.id} with one input propagates {parent_var} as {spider_var}\n\t" + line
    #                     function_result_variables[node.id] = spider_var
    #                 else:
    #                     spider_var = f"res_{result_index}"
    #                     result_index += 1
    #                     inputs_list = []
    #                     for wire in node.inputs:
    #                         input_node = hypergraph.get_node_by_output(wire)
    #                         if input_node is None:
    #                             inputs_list.append(f"input_{input_index}")
    #                             input_index += 1
    #                         else:
    #                             value = function_result_variables[input_node.id]
    #                             if isinstance(value, tuple):
    #                                 inputs_list.append(value[0])
    #                             else:
    #                                 inputs_list.append(value)
    #                     line = f"{spider_var} = [{', '.join(inputs_list)}]\n\t"
    #                     content += f"# Spider node {node.id} aggregates inputs into {spider_var}\n\t" + line
    #                     function_result_variables[node.id] = (spider_var, False)
    #                 continue
    #             else:
    #                 raise ValueError("No box function found for node id: " + str(node.id))
    #
    #         line = f"res_{result_index} = {renamed_functions[current_box_function]}("
    #         result_index += 1
    #         function_result_variables[node.id] = f"res_{result_index - 1}"
    #         for input_wire in node.inputs:
    #             input_node = hypergraph.get_node_by_output(input_wire)
    #             if input_node is None:
    #                 line += f"input_{input_index}, "
    #                 input_index += 1
    #             else:
    #                 var_info = function_result_variables.get(input_node.id)
    #                 if var_info is None:
    #
    #                     input_box = canvas.get_box_by_id(input_node.id)
    #                     if input_box is not None and input_box.sub_diagram is not None:
    #                         print("sub diagram for input node", input_node.id)
    #                         # If it's a container, assume its result variable has been set by recursive generation.
    #                         var_info = function_result_variables.get(input_node.id)
    #                         if var_info is None:
    #                             raise ValueError(
    #                                 f"Expected result for container node id {input_node.id} not generated!")
    #                     else:
    #                         raise ValueError(f"Expected result for node id {input_node.id} not found!")
    #
    #                 if isinstance(var_info, tuple):
    #                     var, is_single = var_info
    #                     if is_single:
    #                         line += f"{var}, "
    #                     else:
    #                         line += f"{var}[{function_output_index[input_node.id]}], "
    #                         function_output_index[input_node.id] += 1
    #                 else:
    #                     line += f"{var_info}, "
    #         line = line[:-2] + ")\n\t"
    #         content += line
    #
    #     return content, function_result_variables

    @classmethod
    def create_main_function_content_from_sorted_list(cls, canvas: CustomCanvas, sorted_nodes: List,
                                                      renamed_functions: dict[BoxFunction, str],
                                                      hypergraph: Hypergraph) -> Tuple[str, dict]:
        content, func_vars, _, _ = cls._create_main_function_content_rec(canvas, sorted_nodes,
                                                                         renamed_functions, hypergraph,
                                                                         input_index=1, result_index=1,
                                                                         default_input=None)
        return content, func_vars

    @classmethod
    def _create_main_function_content_rec(cls, canvas: CustomCanvas, sorted_nodes: List,
                                          renamed_functions: dict[BoxFunction, str],
                                          hypergraph: Hypergraph,
                                          input_index: int,
                                          result_index: int,
                                          default_input: Optional[str] = None
                                          ) -> Tuple[str, dict, int, int]:
        function_result_variables: dict[int, str] = {}
        content = ""
        # For nodes with multiple outputs, track the next available index.
        function_output_index: dict[int, int] = {node.id: 0 for node in sorted_nodes if len(node.outputs) > 1}

        for node in sorted_nodes:
            current_box_function = canvas.get_box_function(node.id)
            if current_box_function is None:
                box = canvas.get_box_by_id(node.id)
                if box is not None and box.sub_diagram is not None:
                    print("sub diagram detected for node id:", node.id)
                    sub_hypergraph: Hypergraph = HypergraphManager().get_graph_by_id(box.sub_diagram.id)
                    sub_sorted_nodes = CodeGenerator.topological_sort(sub_hypergraph.nodes, sub_hypergraph)
                    if node.inputs and node.inputs[0] in function_result_variables:
                        container_input = function_result_variables[node.inputs[0]]
                    else:
                        container_input = f"res_{result_index - 1}"

                    # Recursively generate code for the sub-diagram.
                    sub_code, sub_func_vars, input_index, result_index = cls._create_main_function_content_rec(
                        box.sub_diagram, sub_sorted_nodes, renamed_functions, sub_hypergraph,
                        input_index, result_index, container_input
                    )
                    # Append sub-code directly without extra indentation.
                    content += "\n" + sub_code

                    if sub_sorted_nodes:
                        container_output = sub_func_vars[sub_sorted_nodes[-1].id]
                    else:
                        container_output = "None"
                    function_result_variables[node.id] = container_output
                    continue

                if any(spider.id == node.id for spider in canvas.spiders):
                    if len(node.inputs) == 1:
                        spider_var = f"res_{result_index}"
                        result_index += 1
                        if node.inputs:
                            parent_var = function_result_variables.get(
                                node.inputs[0], default_input if default_input is not None else f"input_{input_index}"
                            )
                            if node.inputs[0] not in function_result_variables and default_input is None:
                                input_index += 1
                            else:
                                default_input = None
                        else:
                            parent_var = default_input if default_input is not None else f"input_{input_index}"
                            if default_input is None:
                                input_index += 1
                            else:
                                default_input = None
                        line = f"res_{result_index - 1} = {parent_var}\n"
                        content += f"# Spider node {node.id} with one input propagates {parent_var} as res_{result_index - 1}\n" + line
                        function_result_variables[node.id] = f"res_{result_index - 1}"
                    else:
                        spider_var = f"res_{result_index}"
                        result_index += 1
                        inputs_list = []
                        for wire in node.inputs:
                            input_node = hypergraph.get_node_by_output(wire)
                            if input_node is None:
                                if default_input is not None:
                                    inputs_list.append(default_input)
                                    default_input = None
                                else:
                                    inputs_list.append(f"input_{input_index}")
                                    input_index += 1
                            else:
                                value = function_result_variables[input_node.id]
                                if isinstance(value, tuple):
                                    inputs_list.append(value[0])
                                else:
                                    inputs_list.append(value)
                        line = f"{spider_var} = [{', '.join(inputs_list)}]\n"
                        content += f"# Spider node {node.id} aggregates inputs into {spider_var}\n" + line
                        function_result_variables[node.id] = (spider_var, False)
                    continue
                else:
                    raise ValueError("No box function found for node id: " + str(node.id))

            # Process node with a valid box function.
            line = f"res_{result_index} = {renamed_functions[current_box_function]}("
            result_index += 1
            function_result_variables[node.id] = f"res_{result_index - 1}"
            for input_wire in node.inputs:
                input_node = hypergraph.get_node_by_output(input_wire)
                if input_node is None:
                    if default_input is not None:
                        line += f"{default_input}, "
                        default_input = None
                    else:
                        line += f"input_{input_index}, "
                        input_index += 1
                else:
                    var_info = function_result_variables.get(input_node.id)
                    if var_info is None:
                        input_box = canvas.get_box_by_id(input_node.id)
                        if input_box is not None and input_box.sub_diagram is not None:
                            var_info = function_result_variables.get(input_node.id)
                            if var_info is None:
                                raise ValueError(
                                    f"Expected result for container node id {input_node.id} not generated!")
                        else:
                            raise ValueError(f"Expected result for node id {input_node.id} not found!")
                    if isinstance(var_info, tuple):
                        var, is_single = var_info
                        if is_single:
                            line += f"{var}, "
                        else:
                            line += f"{var}[{function_output_index[input_node.id]}], "
                            function_output_index[input_node.id] += 1
                    else:
                        line += f"{var_info}, "
            line = line[:-2] + ")\n"
            content += line

        return content, function_result_variables, input_index, result_index

    @classmethod
    def create_definition_of_main_function(cls, input_nodes: set[Node]) -> str:

        definition = "def main("
        variables_count = sum(map(lambda node: len(node.inputs), input_nodes))
        has_args = False

        for i in range(variables_count):
            definition += f"input_{i + 1} = None, "
            has_args = True
        definition = (definition[:-2] if has_args else definition) + "):\n\t"

        return definition

    @classmethod
    def create_main_function_content(cls, canvas: CustomCanvas, nodes_queue: Queue[Node],
                                     renamed_functions: dict[BoxFunction, str], hypergraph: Hypergraph
                                     ) -> list[str, dict[int, str]]:

        function_result_variables: dict[int, str] = dict()
        input_index = 1
        result_index = 1
        content = ""

        function_output_index: dict[int, int] = dict()
        for node in hypergraph.nodes:
            if len(node.outputs) > 1:
                function_output_index[node.id] = 0

        while not nodes_queue.empty():
            node = nodes_queue.get()
            variable_name = f"res_{result_index}"

            current_box_function = canvas.get_box_function(node.id)

            if current_box_function is None:
                if any(spider.id == node.id for spider in canvas.spiders):
                    if len(node.inputs) == 1:
                        spider_var = f"res_{result_index}"
                        result_index += 1
                        if node.inputs:
                            parent_var = function_result_variables.get(node.inputs[0], f"input_{input_index}")
                            if node.inputs[0] not in function_result_variables:
                                input_index += 1
                        else:
                            parent_var = f"input_{input_index}"
                            input_index += 1
                        line = f"{spider_var} = {parent_var}\n\t"
                        content += f"# Spider node {node.id} with one input propagates {parent_var} as {spider_var}\n\t" + line
                        function_result_variables[node.id] = spider_var
                    else:
                        spider_var = f"res_{result_index}"
                        result_index += 1
                        inputs_list = []
                        for wire in node.inputs:
                            input_node = hypergraph.get_node_by_output(wire)
                            if input_node is None:
                                inputs_list.append(f"input_{input_index}")
                                input_index += 1
                            else:
                                value = function_result_variables[input_node.id]
                                if isinstance(value, tuple):
                                    inputs_list.append(value[0])
                                else:
                                    inputs_list.append(value)
                        line = f"{spider_var} = [{', '.join(inputs_list)}]\n\t"
                        content += f"# Spider node {node.id} aggregates inputs into {spider_var}\n\t" + line
                        function_result_variables[node.id] = (spider_var, False)
                    continue
                else:
                    raise ValueError("No box function found for node id: " + str(node.id))

            line = f"{variable_name} = {renamed_functions[current_box_function]}("
            result_index += 1
            function_result_variables[node.id] = variable_name

            for input_wire in node.inputs:
                input_node = hypergraph.get_node_by_output(input_wire)
                if input_node is None:
                    line += f"input_{input_index}, "
                    input_index += 1
                else:
                    var_info = function_result_variables[input_node.id]
                    if isinstance(var_info, tuple):
                        var, is_single = var_info
                        if is_single:
                            line += f"{var}, "
                        else:
                            line += f"{var}[{function_output_index[input_node.id]}], "
                            function_output_index[input_node.id] += 1
                    else:
                        line += f"{var_info}, "
            line = line[:-2] + ")\n\t"
            content += line

        return content, function_result_variables

    @classmethod
    def create_main_function_return(cls, function_result_variables: dict[int, str], hypergraph: Hypergraph) -> str:
        return_statement = "return "
        output_nodes = set(hypergraph.get_node_by_output(output) for output in hypergraph.outputs)
        for output_node in output_nodes:
            return_statement += f'{function_result_variables.get(output_node.id)}, '
        return return_statement[:-2]

    @classmethod
    def get_children_nodes(cls, current_level_nodes: list[Node], node_input_count_check: dict[int, int]) -> set:
        children = set()

        for node in current_level_nodes:
            current_node_children = node.get_children()

            for node_child in current_node_children:
                connections_with_parent_node = 0
                for parent_node_output in node.outputs:
                    if parent_node_output in node_child.inputs:
                        connections_with_parent_node += 1
                        
                node_input_count_check[node_child.id] = node_input_count_check.get(node_child.id, 0) + connections_with_parent_node

                if node_input_count_check[node_child.id] == len(node_child.inputs):
                    children.add(node_child)

        return children

    @staticmethod
    def get_item_y(canvas: CustomCanvas, node_id: int) -> float:
        box = canvas.get_box_by_id(node_id)
        if box is not None:
            return box.y
        for spider in canvas.spiders:
            if spider.id == node_id:
                return spider.y
        return 0.0
