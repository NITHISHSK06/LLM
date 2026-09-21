# Rivuni LLM Chatbot

Rivuni is a local chatbot UI for the project's real Exp012 decoder-only model.

```text
React UI
   |
FastAPI
   |
model_service.py
   |
Exp012 25.52M Transformer
   |
tokenizer_exp003
   |
Generated response
```

## Installation

From the project root, install the Python dependencies in the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm install
```

## Start the backend

From the project root:

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

The backend loads the tokenizer and instruction-tuned checkpoint once during startup. It selects CUDA when available and otherwise uses CPU. The default checkpoint is `checkpoints/exp012_capacity_25m_sft/best_model.pt`, initialized from the Exp012 base model.

## Start the frontend

In a second terminal:

```powershell
cd frontend
npm run dev
```

The Vite development server normally runs at `http://localhost:5173`.

Set `frontend/.env` when the API is hosted elsewhere:

```text
VITE_API_URL=http://localhost:8000
```

The backend also accepts these optional environment variables:

```text
EXP012_CHECKPOINT=D:\path\to\best_model.pt
EXP012_TOKENIZER=D:\path\to\tokenizer_exp003
```

## API

### `GET /api/health`

Example response:

```json
{
  "status": "ok",
  "model": "Exp012",
  "parameters": 25523712,
  "device": "cuda"
}
```

### `POST /api/chat`

```json
{
  "message": "Hello, how are you?",
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.9,
  "max_new_tokens": 80
}
```

The response contains only the assistant continuation:

```json
{
  "response": "...",
  "model": "Exp012",
  "parameters": 25523712
}
```

Generation uses the existing instruction format `User: {message}\nAssistant: ` and does not send previous local chat history to the model. Chat history is stored in browser `localStorage` only.

## Manual integration test

1. Start the backend and wait for `Model ready.` in its log.
2. Open `http://localhost:8000/api/health` and confirm `status` is `ok` and `model` is `Exp012`.
3. Start the frontend with `npm run dev`.
4. Send `Hello` from the UI.
5. Confirm the assistant message is returned by `POST /api/chat` and that the response metadata says `Exp012`.

## Troubleshooting

- **LLM Offline:** confirm the backend is running on port 8000 and inspect its startup log for a missing checkpoint or tokenizer file.
- **Checkpoint not found:** set `EXP012_CHECKPOINT` to a valid `.pt` file containing `model_config` and `model_state_dict`.
- **Slow generation:** CPU inference is supported but substantially slower than CUDA. Reduce max new tokens in Settings.
- **CUDA unavailable:** this is expected when PyTorch cannot access a compatible GPU; the service automatically uses CPU.
- **Port already in use:** start Uvicorn on another port and set `VITE_API_URL` to that port before starting Vite.