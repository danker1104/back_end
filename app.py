import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="대한민국 법률 AI 챗봇", version="1.0.0")


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


class ChatResponse(BaseModel):
    answer: str


def _get_chat_runners() -> tuple[Any, Any]:
    from chat import run_chat, run_chat_v2

    return run_chat, run_chat_v2


@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "Legal chatbot API is running", "status": "ok"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query는 비어 있을 수 없습니다.")

    try:
        run_chat, _ = _get_chat_runners()
        answer = run_chat(request.query, top_k=request.top_k)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="챗봇 백엔드를 불러올 수 없습니다.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(answer=answer)


@app.post("/chat/v2", response_model=ChatResponse)
def chat_v2(request: ChatRequest) -> ChatResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query는 비어 있을 수 없습니다.")

    try:
        _, run_chat_v2 = _get_chat_runners()
        answer = run_chat_v2(request.query, top_k=request.top_k)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="챗봇 백엔드를 불러올 수 없습니다.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(answer=answer)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
