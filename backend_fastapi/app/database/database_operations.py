import os
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Sequence
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, select, literal
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import VECTOR
from dotenv import load_dotenv
from ..utils.embedding_manager import EmbeddingManager
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import time
from ..Agent.logger import logger

# Load environment variables
load_dotenv(override=True)
Database_URL = os.getenv("DATABASE_URL")
if not Database_URL:
    raise ValueError("DATABASE_URL is not set in environment variables")

# Setup SQLAlchemy async engine and sessionmaker
engine: AsyncEngine = create_async_engine(Database_URL, echo=True)
async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)
Base = declarative_base()


# --- ORM Models ---


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Message(Base):
    __tablename__ = "messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("users.id"),
                     nullable=False, index=True)
    sender = Column(String, nullable=False)  # e.g., 'human', 'ai', 'tool'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class SessionLog(Base):
    __tablename__ = "sessions"
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    event_type = Column(String(10), nullable=False)  # 'login' or 'logout'
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class ConversationMemory(Base):
    __tablename__ = "conversation_memory"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, nullable=False, index=True)
    message_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(VECTOR(384))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, nullable=False, index=True)
    summary = Column(Text, nullable=False)
    embedding = Column(VECTOR(384))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


# --- Initialize tables ---


async def check_db_connection(max_retries: int = 3, retry_delay: int = 5) -> bool:
    """
    Check database connection and retry if not available.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds

    Returns:
        bool: True if connection successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                logger.info("✅ Database connection successful!")
                return True
        except SQLAlchemyError as e:
            logger.error(
                f"Database connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            continue
    return False


async def ensure_database_setup() -> bool:
    """
    Ensure database is connected and initialized.

    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Check connection
        if not await check_db_connection():
            logger.error("❌ Could not establish database connection")
            return False

        # Initialize tables
        async with engine.begin() as conn:
            # Check if tables exist
            result = await conn.execute(text(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'users'
                )
                """
            ))
            tables_exist = result.scalar()

            if not tables_exist:
                logger.info("Creating database tables...")
                await conn.run_sync(Base.metadata.create_all)
                logger.info("✅ Database tables created successfully!")
            else:
                logger.info("✅ Database tables already exist")

        return True

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


async def init_models() -> bool:
    """
    Initialize database connection and create tables if needed.

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        if await ensure_database_setup():
            logger.info("✅ Database initialized successfully!")
            return True
        return False
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


# --- Database operations ---


async def add_user(email: str, name: str, password_hash: str):
    """
    Add a new user to the database.

    Returns:
        The new user's ID.
    """
    async with async_session() as session:
        async with session.begin():
            new_user = User(email=email, name=name,
                            password_hash=password_hash)
            session.add(new_user)
            await session.flush()  # ensure new_user.id is generated
            return new_user.id


async def get_user_by_email(email: str) -> Optional[User]:
    """
    Fetch a user by email.

    Returns:
        User instance if found, else None.
    """
    async with async_session() as session:
        result = await session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()


async def save_message(user_id: int, sender: str, content: str) -> None:
    """
    Save a message sent by a user or AI/tool.

    Args:
        user_id: ID of the user.
        sender: 'human', 'ai', or 'tool'.
        content: Message text.
    """
    async with async_session() as session:
        async with session.begin():
            message = Message(user_id=user_id, sender=sender, content=content)
            session.add(message)


async def get_user_messages(user_id: int) -> Sequence[Message]:
    """
    Retrieve all messages for a given user, ordered by timestamp.

    Returns:
        List of Message instances.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Message).where(Message.user_id ==
                                  user_id).order_by(Message.timestamp)
        )
        return result.scalars().all()


async def save_conversation_memory(user_id: int, message_type: str, content: str):
    """
    Save conversation memory entry with embedding vector.

    Args:
        user_id: ID of the user.
        message_type: Type of message (e.g., 'human', 'ai', 'tool').
        content: The message content.
    """
    try:
        embedding_manager = EmbeddingManager.get_instance()
        embedding = embedding_manager.encode(content).tolist()
        async with async_session() as session:
            async with session.begin():
                entry = ConversationMemory(
                    user_id=user_id,
                    message_type=message_type,
                    content=content,
                    embedding=embedding,
                )
                session.add(entry)
    except Exception as e:
        print(f"Error saving conversation memory: {e}")


async def save_summary_to_database(user_id: int, summary: str) -> None:
    """
    Save conversation summary with embedding vector.

    Args:
        user_id: ID of the user.
        summary: Summary text.
    """
    embedding_manager = EmbeddingManager.get_instance()
    embedding = embedding_manager.encode(summary).tolist()
    async with async_session() as session:
        async with session.begin():
            entry = ConversationSummary(
                user_id=user_id, summary=summary, embedding=embedding)
            session.add(entry)


async def log_session_event(user_id: int, event_type: str) -> None:
    """
    Log a login or logout event for a user.

    Args:
        user_id: ID of the user.
        event_type: Must be 'login' or 'logout'.
    """
    if event_type not in ("login", "logout"):
        raise ValueError("Invalid event_type. Must be 'login' or 'logout'.")

    async with async_session() as session:
        async with session.begin():
            log = SessionLog(user_id=user_id, event_type=event_type)
            session.add(log)


async def search_memory_by_vector(user_id: int, query_vector, top_k: int = 3) -> List[str]:
    """
    Perform a vector similarity search on conversation memory for the given user.

    Args:
        user_id: ID of the user.
        query_vector: Vector to search with (numpy array or list).
        top_k: Number of top results to return.

    Returns:
        List of content strings from the most similar memory entries.
    """
    vec_list = query_vector.tolist()
    async with async_session() as session:
        result = await session.execute(
            select(ConversationMemory.content)
            .where(ConversationMemory.user_id == user_id)
            .order_by(ConversationMemory.embedding.op("<->")(literal(vec_list)))
            .limit(top_k)
        )
        rows = result.fetchall()
        return [row[0] for row in rows]


async def search_summary_by_vector(user_id: int, query_vector, top_k: int = 2) -> List[str]:
    """
    Perform a vector similarity search on conversation summary for the given user.

    Args:
        user_id: ID of the user.
        query_vector: Vector to search with (numpy array or list).
        top_k: Number of top results to return.

    Returns:
        List of content strings from the most similar summary entries.
    """
    vec_list = query_vector.tolist()
    async with async_session() as session:
        result = await session.execute(
            select(ConversationSummary.summary)
            .where(ConversationSummary.user_id == user_id)
            .order_by(ConversationSummary.embedding.op("<->")(literal(vec_list)))
            .limit(top_k)
        )
        rows = result.fetchall()
        return [row[0] for row in rows]


async def get_conversation_memory(user_id: int) -> Sequence[ConversationMemory]:
    """
    Retrieve full conversation memory for a user ordered by timestamp.

    Returns:
        List of ConversationMemory instances.
    """
    async with async_session() as session:
        result = await session.execute(
            select(ConversationMemory).where(ConversationMemory.user_id ==
                                             user_id).order_by(ConversationMemory.timestamp)
        )
        return result.scalars().all()


# --- Run table initialization for testing ---


if __name__ == "__main__":
    async def main():
        if await init_models():
            print("Database setup completed successfully!")
        else:
            print("Database setup failed!")

    asyncio.run(main())
