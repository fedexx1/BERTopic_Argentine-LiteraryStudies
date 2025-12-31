# File: validate_data.py
import os
import json

def validate_data_integrity(articles_folder, metadata_file):
    """
    Checks if the set of .txt files in a folder matches the set of keys in a JSON map.
    """
    print("--- Starting Data Validation ---")
    
    # 1. Get all .txt filenames from the 'articles' folder
    try:
        # We use a 'set' for efficient comparison
        files_in_folder = {f for f in os.listdir(articles_folder) if f.lower().endswith('.txt')}
        print(f"Found {len(files_in_folder)} .txt files in the '{articles_folder}' folder.")
    except FileNotFoundError:
        print(f"Error: The folder '{articles_folder}' was not found. Please make sure it's in the same directory as the script.")
        return

    # 2. Load the metadata map and get all its keys
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_map = json.load(f)
        # We also convert the JSON keys to a 'set'
        keys_in_json = set(metadata_map.keys())
        print(f"Found {len(keys_in_json)} entries in the '{metadata_file}' file.")
    except FileNotFoundError:
        print(f"Error: The metadata file '{metadata_file}' was not found.")
        return

    # 3. Compare the two sets to find any differences
    
    # Files that exist in the folder but are missing from the JSON map
    files_missing_from_json = files_in_folder - keys_in_json
    
    # Entries that exist in the JSON map but have no matching file in the folder
    keys_missing_from_folder = keys_in_json - files_in_folder
    
    # 4. Report the results
    print("-" * 30)
    if not files_missing_from_json and not keys_missing_from_folder:
        print("✅ Success! All checks passed.")
        print("Every .txt file has a corresponding entry in the metadata map, and vice-versa.")
    else:
        print("⚠️ Validation failed. The following discrepancies were found:")
        
        if files_missing_from_json:
            print(f"\n[!] {len(files_missing_from_json)} file(s) exist in the 'articles' folder but are MISSING from the JSON map:")
            for filename in sorted(list(files_missing_from_json)):
                print(f"  - {filename}")
        
        if keys_missing_from_folder:
            print(f"\n[!] {len(keys_missing_from_folder)} entr(y/ies) exist in the JSON map but have NO matching file in the 'articles' folder:")
            for key in sorted(list(keys_missing_from_folder)):
                print(f"  - {key}")

if __name__ == "__main__":
    articles_directory = "./articles"
    metadata_json_file = "metadata_map.json"
    
    validate_data_integrity(articles_directory, metadata_json_file)