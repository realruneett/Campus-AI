import os
import sys
from huggingface_hub import HfApi, login

def deploy_model():
    print("="*60)
    print("  CAMPUS AI - HUGGING FACE DEPLOYMENT")
    print("="*60)
    
    # 1. Ask for credentials and repo ID
    hf_token = input("\nEnter your Hugging Face WRITE Token (paste and press Enter): ").strip()
    repo_id = input("Enter your Hugging Face Repository ID (e.g. your_username/campus-ai-poster-sdxl): ").strip()
    
    if not hf_token or not repo_id:
        print("\n[!] Error: Token and Repository ID are required.")
        sys.exit(1)
        
    try:
        # 2. Authenticate
        print("\n[+] Authenticating with Hugging Face...")
        login(token=hf_token)
        api = HfApi()
        
        # 3. Verify Phase 3 Model exists
        model_dir = "models/sdxl/checkpoints/campus_ai_poster_sdxl_phase3"
        model_file = os.path.join(model_dir, "campus_ai_poster_sdxl_phase3.safetensors")
        
        if not os.path.exists(model_file):
            print(f"\n[!] Error: Phase 3 model not found at {model_file}!")
            print("Make sure Phase 3 training has finished successfully.")
            sys.exit(1)
            
        print("\n[+] Creating/Verifying repository...")
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        
        # 4. Upload the model
        print(f"\n[+] Uploading Phase 3 Model to {repo_id}...")
        api.upload_file(
            path_or_fileobj=model_file,
            path_in_repo="campus_ai_poster_sdxl_phase3.safetensors",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload final Campus AI Phase 3 LoRA weights"
        )
        
        print("\n" + "="*60)
        print(f"  ✅ DEPLOYMENT SUCCESSFUL!")
        print(f"  Model is now live at: https://huggingface.co/{repo_id}")
        print("="*60)
        print("You can now connect this model directly to your Hugging Face space.")
        
    except Exception as e:
        print(f"\n[!] Deployment Failed: {str(e)}")

if __name__ == "__main__":
    deploy_model()
