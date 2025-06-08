from fastapi import APIRouter, HTTPException
from ..database.database_operations import async_session, User, Message
from sqlalchemy import select
from passlib.hash import bcrypt
from ..core.security import create_jwt_token
from pydantic import BaseModel

router = APIRouter()

# --- Models ---


class LoginRequest(BaseModel):
    email: str
    password: str

# --- Endpoints ---


@router.get("/users")
async def get_all_users():
    async with async_session() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        return [{"id": u.id, "name": u.name, "email": u.email, "password_hash": u.password_hash} for u in users]


@router.post("/register")
async def register_user(name: str, email: str, password: str):
    try:
        async with async_session() as session:
            result = await session.execute(select(User).where(User.email == email))
            existing_user = result.scalar_one_or_none()
            if existing_user:
                raise HTTPException(
                    status_code=400, detail="Email already registered")

            hashed_password = bcrypt.hash(password)
            new_user = User(name=name, email=email,
                            password_hash=hashed_password)
            session.add(new_user)
            await session.commit()

            return {"message": "Registration successful"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Registration failed: {e}")


@router.post("/login")
async def login(request: LoginRequest):
    try:
        async with async_session() as session:
            result = await session.execute(select(User).where(User.email == request.email))
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(
                    status_code=401, detail="Invalid email or password")

            if not bcrypt.verify(request.password, str(user.password_hash)):
                raise HTTPException(
                    status_code=401, detail="Invalid email or password")

            token = create_jwt_token(int(user.id))  # type: ignore
            return {"access_token": token}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {e}")


@router.get("/messages")
async def get_user_messages(email: str):
    async with async_session() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        result = await session.execute(
            select(Message).where(Message.user_id ==
                                  user.id).order_by(Message.timestamp)
        )
        messages = result.scalars().all()

        convo = []
        temp_user_msg = ""
        for msg in messages:
            if str(msg.sender) == "human":
                temp_user_msg = msg.content
            elif str(msg.sender) == "ai":
                convo.append(
                    {"user_msg": temp_user_msg, "ai_msg": msg.content})

        return convo
