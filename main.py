from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from agent import agent


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    question: str
    company_name: Optional[str] = None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Botivate RAG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h3>index.html not found</h3>",
            status_code=404
        )


@app.head("/")
async def status_check():
    return Response(status_code=200)


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    Main chat endpoint
    """

    initial_state = {
        "question": request.question,
        "company_name": request.company_name,
        "answer": ""
    }

    try:
        final_state = agent.invoke(initial_state)
        return {
            "answer": final_state.get(
                "answer",
                "Sorry, I couldn't generate a response."
            )
        }

    except Exception as e:
        print("❌ Agent Error:", e)
        return {
            "answer": (
                "Sorry, something went wrong while processing your request. "
                "Please try again."
            )
        }