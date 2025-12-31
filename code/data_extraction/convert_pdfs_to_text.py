# File: convert_pdfs_to_text.py
# This script should be placed INSIDE the ALL_JOURNALS folder.

import fitz  # PyMuPDF
import os

def convert_pdfs_in_folder(source_folder, output_folder):
    """
    Recursively finds all PDFs in the source_folder, extracts their full text,
    and saves them as .txt files in the output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Starting conversion...")
    print(f"Scanning from: '{source_folder}'")
    print(f"Saving text files to: '{output_folder}'")
    
    converted_count = 0
    
    # Use os.walk to go through all nested folders
    # We will exclude the '.venv' directory from the search.
    for dirpath, dirnames, filenames in os.walk(source_folder):
        # This line prevents the script from searching inside the .venv folder
        if '.venv' in dirnames:
            dirnames.remove('.venv')

        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(dirpath, filename)
                
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(output_folder, txt_filename)
                
                try:
                    doc = fitz.open(pdf_path)
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                    doc.close()
                    
                    with open(txt_path, "w", encoding="utf-8") as txt_file:
                        txt_file.write(full_text)
                    
                    print(f"  -> Converted: {filename}  =>  {txt_filename}")
                    converted_count += 1
                    
                except Exception as e:
                    print(f"  -> ERROR processing {filename}: {e}")

    print(f"\nConversion complete. Total files converted: {converted_count}")

if __name__ == "__main__":
    # The source is the current folder ('.') because the script is inside ALL_JOURNALS.
    source_directory = "."
    
    # The output folder ('../articles') will be created one level UP from the current directory.
    output_directory = "../articles"
    
    convert_pdfs_in_folder(source_directory, output_directory)