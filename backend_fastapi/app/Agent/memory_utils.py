from langchain.prompts import PromptTemplate
from .prompts import SUMMARIZATION_PROMPT
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from .logger import LOGGER as logger


async def summarize_conversation(agent_llm, conversation_messages):
    chat_history = []
    for msg in conversation_messages:
        if isinstance(msg, HumanMessage):
            chat_history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            chat_history.append(f"Assistant: {msg.content}")
        elif isinstance(msg, ToolMessage):
            chat_history.append(
                f"Tool {msg.tool_name}: {msg.content}")  # type: ignore

    history_text = "\n".join(chat_history)
    prompt = PromptTemplate(template=SUMMARIZATION_PROMPT,
                            input_variables=["chat_history"])
    formatted_prompt = prompt.format(chat_history=history_text)

    summary_response = await agent_llm.ainvoke(formatted_prompt)

    logger.info("✅ Short-term memory summarization complete.")

    # Assuming it's AIMessage
    if isinstance(summary_response, AIMessage):
        # Ensure the content is always returned as a string
        if isinstance(summary_response.content, str):
            return summary_response.content
        else:
            return str(summary_response.content)
    else:
        logger.warning("Unexpected summarizer response type.")
        return str(summary_response)
