import os
from inspect import signature


def get_predefined_functions() -> dict:
    predefined_functions = {}
    base_path = os.path.join(os.path.dirname(__file__), "./predefined/")

    directories = [base_path, os.path.join(base_path, "transformer_split_up"),
                   os.path.join(base_path, "transformer_split_up_1"),
                   os.path.join(base_path, "ffn_DSL"),
                   os.path.join(base_path, "cnn_DSL"),
                   os.path.join(base_path, "rnn_DSL"),
                   os.path.join(base_path, "transformer_DSL")]

    for functions_path in directories:
        if not os.path.exists(functions_path):
            continue
        for name in os.listdir(functions_path):
            full_path = os.path.join(functions_path, name)
            if os.path.isfile(full_path) and name.endswith(".py"):
                with open(full_path, "r") as file:
                    function_name = name.replace(".py", "").replace("_", " ")
                    predefined_functions[function_name] = file.read()
    return predefined_functions


functions = get_predefined_functions()


def safe_format(code: str, substitution_dict: dict) -> str:
    """
    Replace only the keys provided in substitution_dict in the code template.
    This avoids processing other curly-brace literals (like those in the meta dictionary).
    """
    for key, value in substitution_dict.items():
        if key == "non_linearity" or key == "pool_type":
            code = code.replace("{" + key + "}", '"' + str(value) + '"')
        else:
            code = code.replace("{" + key + "}", str(value))
    return code


class BoxFunction:
    def __init__(self, name, code=None, substitution_dict=None):
        self.name = name
        if name in functions:
            self.code: str = functions[name]
        elif code is not None:
            self.code: str = code
        else:
            raise ValueError("Should be specified function code or name of predefined function")
        if substitution_dict:
            try:
                self.code = safe_format(self.code, substitution_dict)
            except Exception as e:
                raise ValueError(f"Error formatting code for {name}: {e}")
        local = {}
        exec(self.code, {}, local)
        self.function = local["invoke"]
        self.meta = local["meta"]

    def __call__(self, *args):
        return self.function(*args)

    def count_inputs(self):
        sig = signature(self.code)
        params = sig.parameters
        count = len(params)
        if params["self"]:
            count -= 1
        return count

    def __eq__(self, other):
        if isinstance(other, BoxFunction):
            return self.code == other.code
        return False

    def __hash__(self):
        return hash(self.code)

    def __str__(self):
        return self.name
