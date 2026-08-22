import os
import sys

# Set HF_HOME to the project's local cache folder before importing huggingface_hub
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["HF_HOME"] = os.path.join(project_root, ".hf_cache")

from huggingface_hub import hf_hub_download

def download():
    sys.path.insert(0, project_root)
    from backend.config import settings
    
    model_name = settings.EMBEDDING_MODEL
    repo_name = model_name.split("/")[-1]
    repo_id = f"Xenova/{repo_name}"
    
    print(f"Pre-downloading ONNX model files for {model_name} (repo: {repo_id}) to local cache {os.environ['HF_HOME']}...")
    model_path = hf_hub_download(repo_id=repo_id, filename="onnx/model.onnx")
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    print("Download complete!")
    print(f"Model path: {model_path}")
    print(f"Tokenizer path: {tokenizer_path}")

if __name__ == "__main__":
    download()
