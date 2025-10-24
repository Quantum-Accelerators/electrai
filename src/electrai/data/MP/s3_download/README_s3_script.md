# S3 GGA Data Download Script

This script reads the GGA items from `map_sample.json.gz` and downloads the corresponding files from the Materials Project S3 bucket.

## Prerequisites

1. Install required dependencies:
   ```bash
   pip install -r requirements_s3.txt
   ```

2. No AWS credentials required! The Materials Project S3 bucket is public and accessible without authentication.

## Usage

```bash
python read_gga_from_s3.py
```

## What the script does

1. **Loads map_sample.json.gz**: Reads the compressed JSON file containing the GGA task IDs
2. **Extracts GGA task IDs**: Gets the list of Materials Project task IDs for GGA calculations
3. **Downloads from S3**: Fetches the corresponding `.json.gz` files from `s3://materialsproject-parsed/chgcars/`
4. **Saves locally**: Stores the downloaded files in `./downloaded_chgcars/` directory

## Output

The script will:
- Create a `downloaded_chgcars/` directory
- Download files like `mp-2355719.json.gz`, `mp-1933176.json.gz`, etc.
- Provide logging output showing progress and any errors

## Expected GGA Task IDs

Based on the map_sample.json.gz file, the script will download files for these task IDs:
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
