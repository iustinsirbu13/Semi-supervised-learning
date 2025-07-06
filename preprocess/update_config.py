def update_config_file(file_path, new_values):
    """
    Updates the specified keys in the config file.

    Parameters:
    - file_path: str, path to the config file
    - new_values: dict, keys are config keys to update, values are new values
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        for key, new_val in new_values.items():
            if line.strip().startswith(f"{key}:"):
                line = f"{key}: {new_val}\n"
        updated_lines.append(line)

    with open(file_path, 'w') as f:
        f.writelines(updated_lines)

files = [

]

for file in files:
    update_config_file(
        f"../generated_configs11/{file}.yaml",
        {
            "epoch": 1,
            "save_name": "new_save_name",
            "load_path": "./new/load/path/model.pth",
            "dataset": "new_dataset_name"
        }
    )
