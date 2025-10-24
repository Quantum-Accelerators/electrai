# S3 Materials Project Data Download Script

This script reads task IDs from `map_sample.json.gz` for any specified key and downloads the corresponding files from the Materials Project S3 bucket.

## Prerequisites

Install required dependencies:
   ```bash
   uv pip install -r requirements.txt
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

### Full command line options
```bash
python download_from_s3.py download --help
```

## What the script does

1. **Loads map_sample.json.gz**: Reads the compressed JSON file containing task IDs for various keys
2. **Extracts task IDs**: Gets the list of Materials Project task IDs for the specified key
3. **Downloads from S3**: Fetches the corresponding `.json.gz` files from `s3://materialsproject-parsed/chgcars/`
4. **Saves locally**: Stores the downloaded files in the specified directory

## Output

The script will:
- Create a `downloaded_chgcars/` directory
- Download files like `mp-2355719.json.gz`, `mp-1933176.json.gz`, etc.
- Provide logging output showing progress and any errors

## Command Line Options

### download command
- `--key`: Key to extract from map_sample.json.gz (default: GGA)
- `--map_file`: Path to map_sample.json.gz file (default: ../map/map_sample.json.gz)
- `--output_dir`: Local directory to save downloaded files (default: ./downloaded_chgcars)
- `--bucket`: S3 bucket name (default: materialsproject-parsed)
- `--prefix`: S3 prefix/folder path (default: chgcars)

### list_keys command
- `--map_file`: Path to map_sample.json.gz file (default: ../map/map_sample.json.gz)

## Example GGA Task IDs

Based on the current map_sample.json.gz file, the GGA key contains these task IDs:
- mp-2355719
- mp-1933176
- mp-2507978
- mp-2255579
- mp-1800415
- mp-1923722
- mp-2452291
- mp-1790998
- mp-2632472
- mp-1802556

## Error Handling

The script includes comprehensive error handling and logging:
- Logs successful downloads
- Reports failed downloads with error messages
- Provides summary statistics at the end
