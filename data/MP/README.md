# S3 Materials Project Data Download Script

This script reads task IDs from `chgcars_functional_to_task_ids.json.gz` for any specified key and downloads the corresponding files from the Materials Project S3 bucket.

## Prerequisites

Install required dependencies:
   ```bash
   uv sync
   ```


## Usage

### Basic Usage (downloads GGA data by default)
```bash
python download_from_s3.py download
```

### Download specific key
```bash
python download_from_s3.py download --key=GGA
```

### List available keys
```bash
python download_from_s3.py list_keys
```

### Custom output directory
```bash
python download_from_s3.py download --key=GGA --output_dir=./my_data
```

### Parallel downloads with custom worker count
```bash
# Use 5 parallel workers (default is 10)
python download_from_s3.py download --key=GGA --max_workers=5
```

### Full command line options
```bash
python download_from_s3.py download --help
```

## What the script does

1. **Loads chgcars_functional_to_task_ids.json.gz**: Reads the compressed JSON file containing task IDs for various keys
2. **Extracts task IDs**: Gets the list of Materials Project task IDs for the specified key
3. **Downloads from S3**: Fetches the corresponding `.json.gz` files from `s3://materialsproject-parsed/chgcars/` using parallel processing
4. **Saves locally**: Stores the downloaded files in the specified directory under the corresponding key

## Output

The script will:
- Create a `downloaded_chgcars/` directory
- Download files like `mp-2355719.json.gz`, `mp-1933176.json.gz`, etc.
- Provide logging output showing progress and any errors

## Command Line Options

### download command
- `--key`: Key to extract from chgcars_functional_to_task_ids.json.gz (default: GGA)
- `--map_file`: Path to chgcars_functional_to_task_ids.json.gz file (default: ../map/chgcars_functional_to_task_ids.json.gz)
- `--output_dir`: Local directory to save downloaded files (default: ~/data/MP/downloaded_chgcars)
- `--bucket`: S3 bucket name (default: materialsproject-parsed)
- `--prefix`: S3 prefix/folder path (default: chgcars)
- `--max_workers`: Maximum number of worker threads for parallel downloads (default: 10)

### list_keys command
- `--map_file`: Path to chgcars_functional_to_task_ids.json.gz file (default: ../map/chgcars_functional_to_task_ids.json.gz)

## Example GGA Task IDs

Based on the current map_sample.json.gz file, the GGA key contains these task IDs:
- mp-1775579
- mp-1828106
- mp-1828986
- mp-1887555
- mp-1924667
- mp-2367080
- mp-2411562
- mp-2493198
- mp-2680643
- mp-2709708

The `.json.gz` files for these task IDs are available in `data/MP/jsongz`, and the corresponding VASP CHGCAR files can be found in `data/MP/chgcars`.
The label charge densities where obtained from the Material Project database.

## Error Handling

The script includes comprehensive error handling and logging:
- Logs successful downloads
- Reports failed downloads with error messages
- Provides summary statistics at the end
