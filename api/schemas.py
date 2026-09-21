from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
	message: str = Field(..., min_length=1, max_length=10_000)
	temperature: float = Field(default=0.7, gt=0.0)
	top_k: int = Field(default=40, ge=0)
	top_p: float = Field(default=0.9, gt=0.0, le=1.0)
	max_new_tokens: int = Field(default=80, ge=1, le=256)


class ChatResponse(BaseModel):
	response: str
	model: str
	parameters: int


class HealthResponse(BaseModel):
	status: str
	model: str
	parameters: int
	device: str