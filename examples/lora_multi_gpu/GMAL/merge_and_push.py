import os
# Install transformers
os.system("pip install transformers==4.41.0")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Merge and push adapter to Hugging Face Hub")
parser.add_argument('--adapter_path', type=str, required=True, help="Path to the adapter")
parser.add_argument('--base_model', type=str, required=True, help="Path to the base model on Hugging Face Hub")
parser.add_argument('--token', type=str, required=True, help="Hugging Face token")
parser.add_argument('--org', type=str, default="Ali-C137", help="Organization name or username on Hugging Face Hub")
args = parser.parse_args()

# Define paths and names
ADAPTER_PATH = args.adapter_path
BASE_MODEL = args.base_model
TOKEN = args.token
ORG = args.org
ADAPTER_NAME = ADAPTER_PATH.replace('/', '-').replace('.', '') + "-adapter"
MODEL_NAME = ADAPTER_PATH.replace('/', '-').replace('.', '') + "-model"

# Load the tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# Load the adapter and push it to the hub
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.push_to_hub(f"{ORG}/{ADAPTER_NAME}", private=True, token=TOKEN)

# Merge the adapter with the base model
model = model.merge_and_unload()
model.push_to_hub(f"{ORG}/{MODEL_NAME}", private=True, token=TOKEN)

print("\n########## ALL FINISHED ##########\n")
