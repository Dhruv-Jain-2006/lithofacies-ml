import os
import json
import shutil

def main():
    json_path = 'data/models/preloaded_predictions.json'
    out_dir = 'data/predictions'
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found!")
        return
        
    print(f"Loading giant predictions file: {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} combinations. Creating output directory '{out_dir}'...")
    os.makedirs(out_dir, exist_ok=True)
    
    for key, val in data.items():
        # Key format: {well_id}_{model_name}_{feature_set}
        # Replace spaces in model name with underscores for safe filenames if needed, 
        # but let's keep it clean. Let's replace spaces in filename to be safe, e.g. Random Forest -> Random_Forest
        safe_key = key.replace(' ', '_')
        file_path = os.path.join(out_dir, f"{safe_key}.json")
        
        with open(file_path, 'w') as out_f:
            json.dump(val, out_f)
            
    print(f"Successfully split into {len(data)} individual JSON files inside '{out_dir}'!")
    
    # Optionally delete the giant file to avoid committing it
    print(f"Removing giant temporary file {json_path}...")
    try:
        os.remove(json_path)
        print("Removed successfully!")
    except Exception as e:
        print(f"Could not remove file: {e}")

if __name__ == '__main__':
    main()
