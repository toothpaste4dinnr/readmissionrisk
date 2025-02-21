import os
import sys
from pathlib import Path

# Get the absolute path to the project root directory
root_dir = Path(__file__).parent.absolute()

# Add the project root to Python path
sys.path.append(str(root_dir))

# Change the working directory to the project root
os.chdir(str(root_dir))

# Run the Streamlit app
if __name__ == "__main__":
    os.system(f"streamlit run {str(root_dir)}/app/main.py") 