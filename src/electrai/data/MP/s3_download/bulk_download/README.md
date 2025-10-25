# Dataset

The volumetric data files (CHGCAR, AECCAR0, AECCAR2, ELFCAR) and DOSCAR were downloaded from the Materials Project on 10-22-2025 (database version v2025.09.25)

## Data Generation Steps

To regenerate this data:

0. In a fresh directory, do the following:
1. Run `fetch_deprecated_task_ids.py` and `map_material_to_task_ids.py`
2. Run each of the `download_*.sh` files. This can be done concurrently. It is best to send this to the background with `nohup <script.py> &`
3. Run `download_task_docs.py`
4. Run `remove_deprecated_data.py`
5. Run `remove_missing_task_docs.py`
6. Run `remove_lcharg_false_chgcars.py`
7. Run `remove_nscf_task_ids.py`
8. Run `remove_missing_task_to_material_mapping_chgcars.py`
9. Run `functional_to_task_ids.py`

## Trash

The `trash` folder contains all invalid volumetric data files.

## Metadata

The `metadata` folder contains information about the dataset that is used for filtering:

- `deprecated_material_ids.csv`: A list of all Material IDs that are deprecated according to the Materials Project.
- `deprecated_task_ids.csv`: A list of all Task IDs associated with deprecated Material IDs.
- `material_id_to_task_ids.json.gz`: A mapping of Material IDs to Task IDs.
- `task_id_to_material_id.json.gz`: A mapping of Task IDs to Material IDs.
- `chgcars_functional_to_task_ids.json.gz`: A mapping of functional to chgcars Task IDs.
- `elfcars_functional_to_task_ids.json.gz`: A mapping of functional to elfcars Task IDs.
