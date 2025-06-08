from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from .memory_utils import summarize_conversation
from ..utils.embedding_manager import EmbeddingManager
from ..database.database_operations import (
    save_conversation_memory,
    save_summary_to_database,
    search_memory_by_vector,
    search_summary_by_vector,
)
from .logger import LOGGER as logger


class MemoryManager:
    def __init__(self, llm, agent):
        self.llm = llm
        self.agent = agent
        self.embedding_manager = EmbeddingManager.get_instance()

    async def summarize_if_needed(self, state):
        try:
            if len(state.short_term_memory) > 10:
                logger.info(
                    "Short-term memory limit exceeded. Summarizing memory.")
                summary = await summarize_conversation(self.llm, state.short_term_memory[:-3])
                summary_msg = SystemMessage(content=f"Summary: {summary}")
                state.long_term_summaries.append(summary_msg)

                if state.user_id and summary:
                    logger.info("Saving summary to database...")
                    await save_summary_to_database(state.user_id, summary)

                state.short_term_memory = state.short_term_memory[-3:]
        except Exception as e:
            logger.error(f"Error summarizing memory: {e}")
            pass

    # async def build_context_history(self, state):
    #    query_embedding = self.embedding_manager.encode(state.message)
    #    context_parts = []
#
    #    if state.user_id is not None:
    #        similar_memories = await search_memory_by_vector(int(state.user_id), query_embedding, top_k=3)
    #        if similar_memories:
    #            labeled_memories = '\n'.join(
    #                f"User previously asked: {m}" for m in similar_memories if m)
    #            context_parts.append(f"Similar Memory:\n{labeled_memories}")
#
    #        summaries = await search_summary_by_vector(int(state.user_id), query_embedding, top_k=2)
    #        if summaries:
    #            labeled_summaries = '\n'.join(
    #                f"Summary of previous chat: {s}" for s in summaries)
    #            context_parts.append(f"Summary:\n{labeled_summaries}")
#
    #    recent_messages = []
    #    for msg in state.short_term_memory[-5:]:
    #        if isinstance(msg, HumanMessage):
    #            recent_messages.append(f"Recent: User said: {msg.content}")
    #        elif isinstance(msg, ToolMessage):
    #            recent_messages.append(
#
    #                f"Recent: Tool {msg.tool_name} replied: {msg.content}")# type: ignore
    #        elif isinstance(msg, AIMessage):
    #            recent_messages.append(
    #                f"Recent: Assistant replied: {msg.content}")
    #        elif isinstance(msg, SystemMessage):
    #            recent_messages.append(f"Summary: {msg.content}")
#
    #    context_parts.extend(recent_messages)
#
    #    for summary in state.long_term_summaries:
    #        context_parts.append(f"Summary: {summary.content}")
#
    #    state.context_history = "\n\n".join(context_parts)
    async def build_context_history(self, state):
        logger.info("Building conversation context via vector recall...")
        memories, summaries = await self.agent.get_conversation_context(state.user_id, state.message)

        context_parts = []

        if memories:
            labeled_memories = '\n'.join(
                f"User previously asked: {m}" for m in memories)
            context_parts.append(f"Similar Memory:\n{labeled_memories}")

        if summaries:
            labeled_summaries = '\n'.join(
                f"Summary of previous chat: {s}" for s in summaries)
            context_parts.append(f"Summary:\n{labeled_summaries}")

        # Add recent short-term memory messages (from current session if any)
        recent_messages = []
        for msg in state.short_term_memory[-5:]:
            if isinstance(msg, HumanMessage):
                recent_messages.append(f"Recent: User said: {msg.content}")
            elif isinstance(msg, ToolMessage):
                recent_messages.append(
                    # type: ignore
                    f"Tool {msg.tool_name} replied: {msg.content}")
            elif isinstance(msg, AIMessage):
                recent_messages.append(f"Assistant replied: {msg.content}")
            elif isinstance(msg, SystemMessage):
                recent_messages.append(f"Summary: {msg.content}")

        context_parts.extend(recent_messages)

        for summary in state.long_term_summaries:
            context_parts.append(f"Summary: {summary.content}")

        state.context_history = "\n\n".join(context_parts)
        logger.info("Context history built successfully.")

    async def process_memory_and_context(self, state):
        await self.summarize_if_needed(state)
        await self.build_context_history(state)
