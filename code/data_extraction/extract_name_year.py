import os
import json
import re

def extract_metadata_from_folders(base_folder):
    metadata_map = {}
    print(f"Starting scan of base folder: '{base_folder}'")
    
    if not os.path.isdir(base_folder):
        print(f"Error: Base folder not found at '{base_folder}'")
        return metadata_map

    for journal_name in os.listdir(base_folder):
        journal_path = os.path.join(base_folder, journal_name)
        
        if os.path.isdir(journal_path):
            # We skip the .venv folder
            if journal_name == ".venv":
                continue
                
            print(f"--- Processing Journal: {journal_name} ---")
            
            for dirpath, _, filenames in os.walk(journal_path):
                publication_year = None
                
                match = re.search(r'\((\d{4})\)', dirpath)
                if match:
                    publication_year = int(match.group(1))

                if publication_year:
                    for filename in filenames:
                        if filename.lower().endswith('.pdf'):
                            txt_filename = os.path.splitext(filename)[0] + ".txt"
                            
                            metadata_map[txt_filename] = {
                                "publication_year": publication_year,
                                "journal": journal_name
                            }
    return metadata_map

if __name__ == "__main__":
    # This path now correctly points to the current folder.
    path_to_all_journals = "."
    
    final_map = extract_metadata_from_folders(path_to_all_journals)
    
    if final_map:
        output_filename = "metadata_map.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_map, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ Metadata map created successfully for {len(final_map)} articles.")
        print(f"Results saved to '{output_filename}'.")
    else:
        print("\nNo metadata was extracted. Please check your folder path and structure.")