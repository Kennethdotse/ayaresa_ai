# Ayaresa AI

This repository runs a Gradio app that uses the MedGemma model with optional RAG (document retrieval) and image + text inputs. The instructions below explain how to install required libraries, configure API tokens, and where to place or fetch RAG documents. A Colab notebook that mirrors `app.py` is included and can use a Google Drive folder for the RAG documents.

## Prerequisites

- Python 3.10 or newer (for local runs)
- Google account (for Colab)
- GPU recommended for reasonable inference speed (Colab Pro/Pro+ or local GPU). CPU-only works but is slower.
- Git (optional)

## Installation (local)

1. Create and activate a virtual environment
   - PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Command Prompt:
     ```
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

2. Install requirements
   - If a `requirements.txt` file exists:
     ```bash
     pip install -r requirements.txt
     ```
   - Or install main dependencies:
     ```bash
     pip install gradio transformers datasets torch pillow langchain chromadb huggingface-hub sentence-transformers
     ```
   - Optional (4-bit quantization on compatible GPUs):
     ```bash
     pip install bitsandbytes
     ```

3. (Optional) Install a specific torch build for your CUDA version. See https://pytorch.org.

## Colab usage (recommended if you don't have a local GPU)

The repo includes a Colab notebook that mirrors `app.py`. The notebook can use a Google Drive folder for RAG documents. The Drive path used in the notebook is:

files = '/content/drive/MyDrive/medical/hb_db'

To use the Drive folder in Colab:

1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   Follow the authorization flow.

2. Verify files exist:
   ```bash
   !ls -lah /content/drive/MyDrive/medical/hb_db
   ```

3. Option A — use Drive path directly in notebook:
   In the Colab notebook set (or ensure) the code uses:
   ```python
   files = '/content/drive/MyDrive/medical/hb_db'
   LOCAL_DOCS_PATH = Path(files)
   ```
   The notebook and app will read PDFs/CSV directly from Drive.

4. Option B — copy files into local Colab workspace (faster I/O):
   ```bash
   !mkdir -p /content/hb_db
   !cp -r /content/drive/MyDrive/medical/hb_db/* /content/hb_db/
   ```
   Then set LOCAL_DOCS_PATH = Path('/content/hb_db').

5. Set HF token in Colab:
   ```python
   import os
   os.environ["HF_TOKEN"] = "hf_XXXXXXXXXXXXXXXXXXXXXXXX"
   os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
   ```

## Environment variables / API keys

Create a Hugging Face token at https://huggingface.co/settings/tokens. Recommended scopes: `read` (and `inference` if needed). Token format begins with `hf_...`.

Set the token locally (temporary):
- PowerShell:
  ```powershell
  $env:HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXX"
  $env:HUGGINGFACEHUB_API_TOKEN=$env:HF_TOKEN
  ```
- Command Prompt:
  ```
  set HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX
  set HUGGINGFACEHUB_API_TOKEN=%HF_TOKEN%
  ```
- Persist across sessions (Windows):
  ```
  setx HF_TOKEN "hf_XXXXXXXXXXXXXXXXXXXXXXXX"
  setx HUGGINGFACEHUB_API_TOKEN "hf_XXXXXXXXXXXXXXXXXXXXXXXX"
  ```

In Colab, set `os.environ["HF_TOKEN"]` as shown above or use Colab secrets.

### Hugging Face Spaces
If deploying to Spaces, add a Space secret named `HUGGINGFACEHUB_API_TOKEN` (or `HF_TOKEN`) with the token value. Use `iface.launch()` without host/port so Spaces manages TLS.

## Where to get or place RAG documents now

- The repository no longer includes a local `hb_db`. Use the Google Drive folder:
  `/content/drive/MyDrive/medical/hb_db`

- Files expected:
  - PDFs (any .pdf)
  - `Final_Dataset.csv` (optional)
  - Other CSVs used for RAG

- You can either:
  - Place files directly in your Drive folder above, or
  - Download/copy files into `/content/hb_db` in Colab (see Colab steps).

## Running the app locally

1. Ensure virtualenv is activated and environment variables are set.
2. Ensure `hb_db` contains the required PDFs/CSV (download from Drive or other source).
3. Run:
   ```bash
   python app.py
   ```
4. The Gradio app opens at http://127.0.0.1:7860 by default.

Notes:
- First startup may be slow (model + vectorstore initialization, embedding model download). Use warmup or run once to cache models.
- If using CPU only, set `USE_QUANTIZATION=0` to avoid bitsandbytes issues:
  - PowerShell (session): `$env:USE_QUANTIZATION="0"`

## Common errors and fixes

- 401 / Unauthorized
  - Ensure `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` is set and starts with `hf_`.
  - Verify:
    ```bash
    huggingface-cli whoami
    ```
    or
    ```bash
    curl -H "Authorization: Bearer $env:HF_TOKEN" https://huggingface.co/api/whoami-v2
    ```

- Mixed content / HTTPS blocked assets (on Spaces)
  - Use `iface.launch()` without host/port arguments.

- Slow responses
  - Use GPU (Colab Pro/Pro+ or local GPU).
  - Reduce model output length or number of retrieved documents (k) in the code.
  - Disable quantization on CPU.

## Security and privacy

- Do not commit secrets to the repo.
- Check privacy/compliance before uploading sensitive medical data.

## Support / Requesting tokens

- Request private-model access from the model owner if needed.
- Create your own Hugging Face token at https://huggingface.co/settings/tokens.

## Next steps

- Add `requirements.txt` to pin versions.
- Add a `.env.example` (no secrets) listing required env var names.
- Document the exact Colab notebook filename and include example cells to fetch or sync the Drive folder.