import sys
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from .Agent.Agent_state import GeminiAgentState
from .Agent.Agentmain import Agent
from .core.security import verify_jwt_token
from .database.database_operations import init_models,  get_conversation_memory
from .api import routers_users

app = FastAPI()
agent = Agent()

# Include user routes
app.include_router(routers_users.router)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str
    sources: list = []


def convert_db_messages_to_langchain(messages):
    history = []
    for m in messages:
        if m.message_type == 'human':
            history.append(HumanMessage(content=m.content))
        elif m.message_type == 'ai':
            history.append(AIMessage(content=m.content))
        elif m.message_type == 'tool':
            history.append(ToolMessage(content=m.content,
                           tool_name="tool", tool_call_id="recalled"))
        elif m.message_type == 'summary':
            history.append(SystemMessage(content=m.content))
    return history


@app.post("/generate", response_model=QueryResponse)
async def generate_response(
    request: Request,
    query_request: QueryRequest,
    user_id: int = Depends(verify_jwt_token)
):
    print("\n=== Request Debug ===")
    print("Headers:", dict(request.headers))
    print("Authorization:", request.headers.get("authorization"))
    sys.stdout.flush()

    try:
        # Fetch and convert conversation history from DB
        messages = await get_conversation_memory(user_id)
        short_term_memory = convert_db_messages_to_langchain(messages)

        # Create agent state
        state = GeminiAgentState(
            message=query_request.query,
            user_id=user_id,
            short_term_memory=short_term_memory
        )

        print("\n=== State Debug ===")
        print(f"User ID: {user_id}")
        print(f"Query: {query_request.query}")
        sys.stdout.flush()

        result_state = await agent.graph.ainvoke(state)

        response_obj = result_state.get("response")
        final_answer = (
            response_obj if isinstance(response_obj, str)
            else getattr(response_obj, "response", None)
            or result_state.get("final_answer")
            or str(response_obj)  # fallback only if nothing else works
            or "I'm sorry, I couldn't generate a response."
        )
        sources = result_state.get("sources") or []

        return {
            "response": final_answer,
            "sources": sources if isinstance(sources, list) else []
        }

    except Exception as e:
        error_msg = f"Agent error: {str(e)}"
        print(f"\n=== Error ===\n{error_msg}")
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/")
def health_check():
    return {"status": "Backend running ✅"}


@app.on_event("startup")
async def startup_event():
    """Initialize database on app startup"""
    try:
        await init_models()
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        raise
