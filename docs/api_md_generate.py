import inspect
import re

from api_md_CONFIG import accessors, ROOT_GITHUB_URL, PACKAGE_NAME

with open("docs/api/accessors.md", "w") as accessors_file:
    accessors_file.writelines("# Accessors API\n")
    accessors_file.writelines("\n")
    for accessor_group, accessors in accessors.items():
        accessors_file.writelines(f"## {accessor_group} Methods\n")
        for accessor in accessors:
            function_name = accessor.__name__
            function_signature = str(inspect.signature(accessor)).replace("pandas.core.frame.DataFrame", "pd.DataFrame")
            docstring = inspect.getdoc(accessor)
            absolute_file_path = inspect.getfile(accessor)
            file_path = re.sub(f".*/{PACKAGE_NAME}/{PACKAGE_NAME}/", "", absolute_file_path)
            github_line_number = inspect.findsource(accessor)[1] + 1
            github_source_url = f"{ROOT_GITHUB_URL}{file_path}#L{github_line_number}"

            accessors_file.writelines(f"#### `{function_name}` *<small>[[source]({github_source_url})]</small>*\n")
            accessors_file.writelines(f"`{function_name}`*{function_signature}*\n")
            accessors_file.writelines("\n")
            accessors_file.writelines(f"{docstring}")
