from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from api.model_service import MODEL_NAME, ModelService
from api.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
service = ModelService()


@asynccontextmanager
async def lifespan(_: FastAPI):
	try:
		service.load()
	except Exception as error:
		logging.getLogger(__name__).exception("Unable to load Exp012")
		raise RuntimeError(f"Unable to load {MODEL_NAME}: {error}") from error
	yield


app = FastAPI(title="Rivuni LLM API", lifespan=lifespan)
app.add_middleware(
	CORSMiddleware,
	allow_origins=[
		"http://localhost:3000",
		"http://127.0.0.1:3000",
		"http://localhost:5173",
		"http://127.0.0.1:5173",
	],
	allow_credentials=True,
	allow_methods=["GET", "POST"],
	allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
	try:
		return HealthResponse(**service.health())
	except RuntimeError as error:
		raise HTTPException(status_code=503, detail="LLM service is not ready") from error


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
	if not request.message.strip():
		raise HTTPException(status_code=422, detail="message must not be empty")
	try:
		response = await run_in_threadpool(
			service.generate,
			request.message,
			request.temperature,
			request.top_k,
			request.top_p,
			request.max_new_tokens,
		)
		return ChatResponse(
			response=response,
			model=MODEL_NAME,
			parameters=service.parameter_count,
		)
	except Exception as error:
		logging.getLogger(__name__).exception("Generation failed")
		raise HTTPException(status_code=500, detail="The LLM could not generate a response") from error