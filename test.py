import os
import zipfile

def zip_first_500_documents(input_folder, output_zip):
    # Get the list of files in the input folder
    documents = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Limit to first 500 documents
    documents = documents[:80]
    
    # Create a zip file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for doc in documents:
            file_path = os.path.join(input_folder, doc)
            zipf.write(file_path, os.path.basename(file_path))
    
    print(f"Zipped {len(documents)} files into {output_zip}")

# Example usage
input_folder = '/Users/nelson/Downloads/Documents (8300_) -  COP29 related/'  # Replace with the path to your folder
output_zip = 'documents.zip'  # Replace with your desired zip file name

zip_first_500_documents(input_folder, output_zip)
