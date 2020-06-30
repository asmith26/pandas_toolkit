import inspect
import re
from typing import Callable

from api_md_CONFIG import accessors, ROOT_GITHUB_URL, PACKAGE_NAME


def get_pretty_function_signature(function: Callable) -> str:
    inputs_and_outputs = str(inspect.signature(function))
    inputs_and_outputs = inputs_and_outputs.replace("pandas.core.frame.DataFrame", "pd.DataFrame")
    inputs_and_outputs = inputs_and_outputs.replace("pandas.core.series.Series", "pd.Series")
    # Keep function output str as is
    inputs, outputs = inputs_and_outputs.split(" -> ")
    # Colour arguments green, types blue
    inputs = inputs.replace(", ", "</span>, <span style='color:green'>")
    inputs = inputs.replace(": ", "</span>: <span style='color:blue'>")
    inputs = inputs.replace(")", "</span>)")
    # Fix self
    inputs = inputs.replace("self</span>,", "self,")
    return f"{inputs} -> {outputs}"


with open("docs/api/accessors.md", "w") as accessors_file:
    accessors_file.writelines("# Accessors API\n")
    accessors_file.writelines("\n")
    for accessor_group, accessors in accessors.items():
        accessors_file.writelines(f"## {accessor_group} Methods\n")
        for accessor in accessors:
            function_name = accessor.__name__
            function_signature = get_pretty_function_signature(accessor)
            docstring = inspect.getdoc(accessor)
            absolute_file_path = inspect.getfile(accessor)
            file_path = re.sub(f".*/{PACKAGE_NAME}/{PACKAGE_NAME}/", "", absolute_file_path)
            github_line_number = inspect.findsource(accessor)[1] + 1
            github_source_url = f"{ROOT_GITHUB_URL}{file_path}#L{github_line_number}"

            accessors_file.writelines(f"### `{function_name}` *<small>[[source]({github_source_url})]</small>*\n")
            accessors_file.writelines(f"`{function_name}`*{function_signature}*\n")
            accessors_file.writelines("\n")
            accessors_file.writelines(f"{docstring}")
            accessors_file.writelines(f"\n\n")
