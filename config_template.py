# Configuration Template for LLM Tagging System
# Copy this file to config.py and fill in your actual values

# OpenAI API Configuration
OPENAI_API_KEY = "your-openai-api-key-here"

# Data Configuration
DATA_FILE_PATH = "your-data-file-path.csv"

# Model Configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.1
BATCH_SIZE = 5

# Analysis Configuration
DEFAULT_SAMPLE_SIZE = 100
RANDOM_STATE = 42

# Output Configuration
OUTPUT_DIR = "output/"
ENABLE_INTERMEDIATE_SAVE = True

# Usage Instructions:
# 1. Copy this file: cp config_template.py config.py
# 2. Edit config.py with your actual values
# 3. Add config.py to .gitignore to keep secrets safe
# 4. Import in your scripts: from config import OPENAI_API_KEY