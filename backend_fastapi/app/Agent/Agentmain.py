import json
import asyncio
import re
import os
import sys
import requests
import uuid
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from ..database.database_operations import get_conversation_memory, save_conversation_memory, save_summary_to_database, search_memory_by_vector, search_summary_by_vector
from .memory_utils import summarize_conversation
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from .logger import LOGGER as logger
from .prompts import SYSTEM_PROMPT, WIKIPEDIA_PROMPT, NEWS_PROMPT, WEBSEARCH_PROMPT, FINAL_RESPONSE_PROMPT
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field, SecretStr
from .memory_manager import MemoryManager
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from .Agent_state import GeminiAgentState, WikiSummaryOutput, NewsSummaryOutput, WebSearchOutput, FinalResponseOutput
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from ..utils.embedding_manager import EmbeddingManager
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))


load_dotenv()  # load .env

groq_api_key = os.getenv("GROQ_API_KEY")
travily_api_key = os.getenv("TAVILY_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")


class StructuredResponse(BaseModel):
    tool: str = Field(
        description="Tool to use: 'wikipedia', 'news', 'websearch', 'end'")
    tool_query: Optional[str] = Field(
        default=None, description="Query to pass to the tool.")
    response: Optional[str] = Field(
        default=None, description="Response text if ready.")
    sources: Optional[List[str]] = Field(
        default=None, description="List of sources or URLs.")


class Agent:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables.")

        # Replace "default-model" with the actual model name
        self._llm = ChatGroq(api_key=SecretStr(groq_api_key),
                             model="llama-3.3-70b-versatile",
                             temperature=1)

        # Placeholder for actual tools you will implement or connect later
        # Initialize tool map with duplicate detection
        self._tool_map = {
            "wikipedia": self.wikipedia_node,
            "news": self.news_summary_node,
            "websearch": self.websearch_node,
            "end": self.final_response_node,
        }
        self.memory_manager = MemoryManager(llm=self._llm, agent=self)
        # Ensure compile_graph returns the compiled graph
        self.graph = self.compile_graph()

        self.embedding_manager = EmbeddingManager.get_instance()

    def compile_graph(self):
        graph_builder = StateGraph(GeminiAgentState)

        # Add all nodes
        graph_builder.add_node("LLM_Decision", self.chat_flow)
        graph_builder.add_node("Wikipedia_Node", self.wikipedia_node)
        graph_builder.add_node("News_Node", self.news_summary_node)
        graph_builder.add_node("Websearch_Node", self.websearch_node)
        graph_builder.add_node("Final_Response_Node", self.final_response_node)
        # graph_builder.add_node("End_Node", END)

        # Set entry point (everything starts at LLM)
        graph_builder.set_entry_point("LLM_Decision")
#
        # Tool routing: LLM decides where to go
        graph_builder.add_conditional_edges("LLM_Decision", self.flow_router, {
            "wikipedia": "Wikipedia_Node",
            "news": "News_Node",
            "websearch": "Websearch_Node",
            "end": "Final_Response_Node",
        })

        # After tool completes — go back to LLM decision
        graph_builder.add_edge("Wikipedia_Node", "LLM_Decision")
        graph_builder.add_edge("News_Node", "LLM_Decision")
        graph_builder.add_edge("Websearch_Node", "LLM_Decision")
        graph_builder.add_edge("Final_Response_Node", END)
#
        graph = graph_builder.compile()
        return graph

    def is_valid_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_map

    # Load conversation memory as LangChain messages
    # async def load_user_memory(self, user_id: int) -> list:
    #    messages = await get_conversation_memory(user_id)
    #    conversation_history = []
    #    for msg in messages:
    #        if str(msg.message_type) == 'human':
    #            conversation_history.append(
    #                HumanMessage(content=str(msg.content)))
    #        elif str(msg.message_type) == 'ai':
    #            conversation_history.append(
    #                AIMessage(content=str(msg.content)))
    #        elif str(msg.message_type) == 'tool':
    #            conversation_history.append(ToolMessage(
    #                content=str(msg.content), tool_name="tool", tool_call_id="loaded"
    #            ))
    #    return conversation_history

    async def get_conversation_context(self, user_id: int, query: str) -> Tuple[list, list]:
        """Get relevant conversation context using vector similarity search."""
        try:
            logger.info("Encoding query for vector recall...")
            query_vector = self.embedding_manager.encode(query)

            memories = await search_memory_by_vector(
                user_id=user_id,
                query_vector=query_vector,
                top_k=3
            )

            summaries = await search_summary_by_vector(
                user_id=user_id,
                query_vector=query_vector,
                top_k=2
            )

            logger.info(
                f"Found {len(memories)} similar memories and {len(summaries)} summaries.")
            return memories, summaries

        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return [], []

    async def get_structured_response(self, output_structure, PROMPT: str, **kwargs):
        parser = PydanticOutputParser(pydantic_object=output_structure)
        template = PromptTemplate(
            template=PROMPT,
            input_variables=list(kwargs.keys()),
            partial_variables={"format_instructions": parser.get_format_instructions()})
        message = template.format(**kwargs)
        llm_structured = self._llm.with_structured_output(output_structure)

        # logger.info(f"Format instructions: {parser.get_format_instructions()}")
        # logger.info(f"Full prompt sent to LLM:\n{message}")

        for attempt in range(1, 4):
            try:
                structured_response = await llm_structured.ainvoke(message)
                # <-- Add this line
                logger.info(
                    f"LLM raw output (attempt {attempt}): {structured_response}")
                if structured_response:
                    return structured_response
                continue
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {str(e)}")
        logger.critical(
            "Failed to get a structured response from the LLM after 3 attempts.")
        raise RuntimeError("Failed to get structured response.")

    async def decide_tool_for_query(self, state: GeminiAgentState) -> GeminiAgentState:
        user_query = state.message

        try:
            parsed = await self.get_structured_response(
                output_structure=StructuredResponse,
                PROMPT=SYSTEM_PROMPT,
                query=user_query,
                context_history=state.context_history or "No previous context"
            )
            logger.info(
                f"Structured LLM output in decide_tool_for_query: {parsed}")
            state.llm_response = str(parsed)
        except Exception as e:
            logger.error(f"LLM decision error: {e}")
            state.next_tool = "end"
            state.error = f"Parsing error: {str(e)}"
            return state

        state.next_tool = getattr(parsed, "tool", "end")
        next_tool = state.next_tool
        if next_tool is None or not self.is_valid_tool(str(next_tool)):
            logger.warning(
                f"Invalid tool chosen by LLM: {next_tool}. Defaulting to 'end'.")
            state.next_tool = "end"
        else:
            # Extract and validate tool query
            tool_query = getattr(parsed, "tool_query", None)
            if not tool_query and state.next_tool != "end":
                logger.warning(
                    f"Tool {state.next_tool} selected without a query. Using original user query.")
                state.tool_query = user_query  # Fallback to original query
            else:
                state.tool_query = tool_query

            # Store response if available
            response = getattr(parsed, "response", None)
            if response:
                state.response = response

            # Store sources for future implementation
            sources = getattr(parsed, "sources", None)
            if sources:
                state.sources = sources

        return state

    async def handle_tool_response(self, state: GeminiAgentState) -> GeminiAgentState:
        tool_name = state.next_tool
        tool_output = getattr(state, "tool_output", "")
        previous_llm_resp = getattr(state, "llm_response", "")

        # Select the correct prompt template for the tool
        if tool_name == "websearch":
            prompt_template = WEBSEARCH_PROMPT
        elif tool_name == "wikipedia":
            prompt_template = WIKIPEDIA_PROMPT
        elif tool_name == "news":
            prompt_template = NEWS_PROMPT
        else:
            logger.error(f"No prompt template found for tool: {tool_name}")
            state.next_tool = "end"
            return state

        # Preprocess tool output if needed
        processed_tool_output = tool_output
        if tool_output and len(tool_output) > 8000:  # If output is very large
            processed_tool_output = tool_output[:7500] + \
                "\n\n[Content truncated due to length...]"
            logger.warning(
                f"Tool output truncated from {len(tool_output)} to 7500 characters")

        # parser = PydanticOutputParser(pydantic_object=StructuredResponse)
        prompt = PromptTemplate(
            template=prompt_template,  # type: ignore
            input_variables=["previous_response", "context"],
            #    partial_variables={
            #        "format_instructions": parser.get_format_instructions()},
        )
        # Use structured output directly rather than going through additional parser
        llm_structured = self._llm.with_structured_output(StructuredResponse)
        message = prompt.format(
            previous_response=previous_llm_resp, context=processed_tool_output)
        logger.info(f"Message to LLM: {message}")
        try:
            llm_response = await llm_structured.ainvoke(message)
            logger.info(f"Structured response from LLM: {llm_response}")
            response_content = llm_response.get("response") if isinstance(
                llm_response, dict) else getattr(llm_response, "response", None)
            sources_content = llm_response.get("sources") if isinstance(
                llm_response, dict) else getattr(llm_response, "sources", None)
            if response_content:
                state.short_term_memory.append(
                    AIMessage(
                        content=response_content,
                        additional_kwargs={
                            "sources": sources_content} if sources_content else {}
                    )
                )
                if state.user_id is not None:
                    await save_conversation_memory(int(state.user_id), 'ai', response_content)
                else:
                    logger.error(
                        "state.user_id is None. Cannot save conversation memory.")
        except Exception as e:
            logger.error(f"LLM response error for tool output: {e}")
            state.next_tool = "end"
            return state

        state.next_tool = getattr(llm_response, "tool", "end")
        state.llm_response = getattr(llm_response, "response", "")
        state.tool_query = getattr(llm_response, "tool_query", None)
        # Store sources directly on state (memory will be implemented later)
        sources = None
        if isinstance(llm_response, dict):
            sources = llm_response.get("sources")
        else:
            sources = getattr(llm_response, "sources", None)
        if sources:
            state.sources = sources

        # Update tool_query for next iteration if continuing
        if state.next_tool != "end":
            tool_query = None
            if isinstance(llm_response, dict):
                tool_query = llm_response.get("tool_query")
            elif hasattr(llm_response, "tool_query"):
                tool_query = getattr(llm_response, "tool_query")
            if tool_query:
                state.tool_query = tool_query
        return state

    async def chat_flow(self, state: GeminiAgentState) -> GeminiAgentState:
        """Handles the main conversation flow with short-term memory, long-term summarization, and DB-backed vector recall."""

        # --- Save new user message to short-term and DB ---
        if state.message:
            state.short_term_memory.append(HumanMessage(content=state.message))
            if state.user_id is not None:
                await save_conversation_memory(int(state.user_id), 'human', state.message)
            else:
                logger.error(
                    "state.user_id is None. Cannot save conversation memory.")
            state.last_message_type = 'human'
        else:
            state.last_message_type = 'ai'  # or 'tool' later if needed

        # --- Initialize next tool and response ---
        try:
            if state.last_message_type == 'human':
                await self.memory_manager.process_memory_and_context(state)

        # --- Summarize if short-term memory exceeds threshold ---
        # if len(state.short_term_memory) > 10:
        #    logger.info(
        #        "Short-term memory limit exceeded. Summarizing memory.")
        #    summary = await summarize_conversation(self._llm, state.short_term_memory[:-3])
        #    summary_msg = SystemMessage(content=f"Summary: {summary}")
        #    if state.user_id is not None:
        #        await save_summary_to_database(int(state.user_id), summary)
        #    else:
        #        logger.error(
        #            "state.user_id is None. Cannot save summary to database.")
        #    state.long_term_summaries.append(summary_msg)
        #    state.short_term_memory = state.short_term_memory[-3:]
#
        # try:
        #    query_embedding = None
        #    context_parts = []
#
        #    # --- Vector recall for memories and summaries ---
        #    if state.message:
        #        query_embedding = self.embedding_manager.encode(state.message)
#
        #        if state.user_id is not None:
        #            # Vector recall from conversation memory
        #            similar_memories = await search_memory_by_vector(int(state.user_id), query_embedding, top_k=3)
        #            if similar_memories:
        #                labeled_memories = '\n'.join(
        #                    f"User previously asked: {m}" for m in similar_memories if m)
        #                context_parts.append(
        #                    f"Similar Memory:\n{labeled_memories}")
#
        #            # Vector recall from conversation summaries
        #            summaries = await search_summary_by_vector(int(state.user_id), query_embedding, top_k=2)
        #            if summaries:
        #                labeled_summaries = '\n'.join(
        #                    f"Summary of previous chat: {s}" for s in summaries)
        #                context_parts.append(f"Summary:\n{labeled_summaries}")
#
        #    # --- Add recent short-term memory (last 5) ---
        #    recent_messages = []
        #    for msg in state.short_term_memory[-5:]:
        #        if isinstance(msg, HumanMessage):
        #            recent_messages.append(f"Recent: User said: {msg.content}")
        #        elif isinstance(msg, ToolMessage):
        #            recent_messages.append(
        #                # type: ignore
        #                f"Recent: Tool {msg.tool_name} replied: {msg.content}")
        #        elif isinstance(msg, AIMessage):
        #            recent_messages.append(
        #                f"Recent: Assistant replied: {msg.content}")
        #        elif isinstance(msg, SystemMessage):
        #            recent_messages.append(f"Summary: {msg.content}")
#
        #    context_parts.extend(recent_messages)
#
        #    # --- Add long-term summaries in memory ---
        #    for summary in state.long_term_summaries:
        #        context_parts.append(f"Summary: {summary.content}")
#
        #    # --- Combine everything into context history ---
        #    state.context_history = "\n\n".join(context_parts)

            # --- Tool response handler or tool decision ---
            try:
                if hasattr(state, "tool_output") and state.tool_output:
                    state = await self.handle_tool_response(state)
                    logger.info(f"Handled tool response for {state.next_tool}")
                else:
                    state = await self.decide_tool_for_query(state)
                    logger.info(f"Decided next tool: {state.next_tool}")

            except Exception as e:
                logger.error("Error in chat flow decision phase: %s", e)
                state.error = str(e)
                state.next_tool = "end"

            # --- Save tool output if any ---
            if state.tool_output:
                tool_msg = ToolMessage(
                    content=state.tool_output[:500],
                    tool_name=state.next_tool or "unknown",
                    tool_call_id=str(uuid.uuid4())
                )
                state.short_term_memory.append(tool_msg)
                if state.user_id is not None:
                    await save_conversation_memory(int(state.user_id), 'tool', state.tool_output)
                else:
                    logger.error(
                        "state.user_id is None. Cannot save conversation memory.")

            # --- Save AI response if available ---
            if state.response:
                ai_msg = AIMessage(content=state.response)
                state.short_term_memory.append(ai_msg)
                if state.user_id is not None:
                    await save_conversation_memory(int(state.user_id), 'ai', state.response)
                else:
                    logger.error(
                        "state.user_id is None. Cannot save conversation memory.")

        except Exception as e:
            logger.error("Error in chat flow execution: %s", e)
            state.error = str(e)
            state.next_tool = "end"

        return state

    def flow_router(self, state: GeminiAgentState) -> str:
        """Router: reads state.next_tool to pick next node."""
        return state.next_tool or "end"

    # wikipedia node
    async def wikipedia_node(self, state: GeminiAgentState) -> GeminiAgentState:
        if not state.tool_query:
            logger.warning("No tool_query provided for Wikipedia search.")
            state.tool_output = "No query provided to Wikipedia tool."
            return state

        logger.info("Entered Wikipedia Search Node")

        try:
            wiki_wrapper = WikipediaAPIWrapper(
                wiki_client=None,  # Replace None with an actual client if needed
                top_k_results=1, doc_content_chars_max=2000)
            # Uncomment the following import if not already imported at the top
            # from langchain.tools import WikipediaQueryRun
            wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

            query = state.tool_query
            result = await wiki_tool.ainvoke(query)

            structured_result = WikiSummaryOutput(
                summary=result,
                page_title=None,  # Optional: You can enhance this if the API gives page titles
                url=None         # Optional: Likewise, if available
            )
            logger.info(f"Structured Wikipedia summary: {structured_result}")

            state.tool_output = structured_result.summary
            logger.info(
                f"Wikipedia summary retrieved: {structured_result.summary}")

            state.short_term_memory.append(
                ToolMessage(
                    content=state.tool_output[:500] if state.tool_output else "",
                    tool_name=state.next_tool or "Wikipedia",
                    tool_call_id=str(uuid.uuid4())
                    # additional_kwargs={"tool_name": "websearch"}
                )
            )

        except Exception as e:
            logger.error(f"Error fetching from Wikipedia: {e}")
            state.tool_output = f"Failed to fetch Wikipedia summary: {str(e)}"

        return state
    # news node

    async def news_summary_node(self, state: GeminiAgentState) -> GeminiAgentState:
        if not state.tool_query:
            logger.warning("No tool_query provided for News Summary search.")
            state.tool_output = "No query provided to News Summary tool."
            return state

        logger.info("Entered News Summary Node")

        try:
            if not travily_api_key:
                logger.error(
                    "TAVILY_API_KEY not found in environment variables")
                raise ValueError(
                    "TAVILY_API_KEY not found in environment variables")

            tavily_wrapper = TavilySearchAPIWrapper(
                tavily_api_key=SecretStr(travily_api_key)
            )
            tavily_tool = TavilySearchResults(api_wrapper=tavily_wrapper)

            # Get raw results from Tavily
            results = await tavily_tool.ainvoke(state.tool_query)

            if isinstance(results, list) and results:
                # Extract content and URLs
                content_list = []
                urls = []

                for item in results:
                    if isinstance(item, dict):
                        if 'content' in item:
                            content_list.append(item['content'])
                        if 'url' in item:
                            urls.append(item['url'])

                # Combine all content into one string
                combined_content = "\n".join(
                    content_list) if content_list else "No content found"

                # Create structured output like Wikipedia node
                structured_result = NewsSummaryOutput(
                    summary=combined_content,
                    sources=urls if urls else None
                )
                logger.info(f"Structured News summary: {structured_result}")

                # Store the summary in tool_output
                state.tool_output = structured_result.summary
                logger.info(
                    f"News summary retrieved: {len(structured_result.summary)} chars")

                state.short_term_memory.append(
                    ToolMessage(
                        content=state.tool_output[:500] if state.tool_output else "",
                        tool_name=state.next_tool or "News",
                        tool_call_id=str(uuid.uuid4())
                        # additional_kwargs={"tool_name": "news"}
                    )
                )
                # Store URLs as sources
                if urls:
                    state.sources = urls

            else:
                logger.warning(
                    "No news results found or invalid response format")
                state.tool_output = "No relevant news found for the query."

        except Exception as e:
            logger.error(f"Error fetching from News: {e}")
            state.tool_output = f"Failed to fetch News summary: {str(e)}"

        logger.info("Exiting News Summary Node")
        return state

    # web_search node
    async def websearch_node(self, state: GeminiAgentState) -> GeminiAgentState:
        """
        Simple, straightforward web search node that works with SerpAPI.
        """
        if not state.tool_query:
            logger.warning("No tool_query provided for Web Search.")
            state.tool_output = "No query provided to Web Search tool."
            return state

        logger.info("Entered Web Search Node")

        try:
            # Check for API key
            serpapi_api_key = os.getenv("SERPAPI_API_KEY")
            if not serpapi_api_key:
                raise ValueError(
                    "SERPAPI_API_KEY not found in environment variables.")
            query = state.tool_query
            params = {
                "engine": "google",
                "q": query,
                "api_key": serpapi_api_key,
                "gl": "us",  # Country (United States)
                "hl": "en"   # Language (English)
            }

            # Send request
            response = requests.get(
                "https://serpapi.com/search",
                params=params,
                timeout=10
            )
            response.raise_for_status()

            # Parse response
            results = response.json()

            # Get organic results (using the correct key we discovered)
            organic_results = results.get("organic_results", [])

            # Extract snippets
            search_results = [
                item.get("snippet", "")
                for item in organic_results
                if item.get("snippet")
            ]

            # Provide results
            if search_results:
                state.tool_output = "\n".join(search_results)
                logger.info(f"Retrieved {len(search_results)} search results")
            else:
                state.tool_output = "No search results found."
                logger.warning("No search results found")

            # Add tool output as ToolMessage to conversation history
            state.short_term_memory.append(
                ToolMessage(
                    content=state.tool_output[:500] if state.tool_output else "",
                    tool_name=state.next_tool or "websearch",
                    tool_call_id=str(uuid.uuid4())
                    # additional_kwargs={"tool_name": "websearch"}
                )
            )

            logger.info("Web Search completed successfully")

        except Exception as e:
            logger.error(f"Error fetching from Web Search: {str(e)}")
            state.tool_output = f"Failed to fetch web search results: {str(e)}"

        return state
    # final response node

    async def final_response_node(self, state: GeminiAgentState) -> GeminiAgentState:
        """
        Final response node that synthesizes information from all tools used
        and generates a comprehensive answer.
        """
        logger.info("Final response node reached.")

        # If we already have a final response from the LLM, use it
        if state.llm_response:
            logger.info(
                f"Using LLM-provided final response: {state.llm_response}")
            state.final_answer = state.llm_response
            return state

        # Otherwise, we need to generate a final response by looking at all collected information
        try:
            # Gather all the context/information collected so far
            context_elements = []

            # Add all tool outputs if they exist
            if hasattr(state, "tool_history") and state.tool_history:
                for tool_entry in state.tool_history:
                    if tool_entry.get("output"):
                        context_elements.append(
                            f"Information from {tool_entry['tool']}: {tool_entry['output']}")

            # If tool_history doesn't exist, try to get the most recent tool_output
            elif hasattr(state, "tool_output") and state.tool_output:
                context_elements.append(
                    f"Information gathered: {state.tool_output}")

            # Combine all context into a single string
            if context_elements:
                combined_context = "\n\n".join(context_elements)
            else:
                combined_context = "No additional information was gathered."

            # Get sources if available
            sources = getattr(state, "sources", [])
            sources_text = "\n".join(
                [f"- {source}" for source in sources]) if sources else "No specific sources available."

            llm_structured = self._llm.with_structured_output(
                FinalResponseOutput)
            # Prepare the final response prompt
            prompt = PromptTemplate(
                template=FINAL_RESPONSE_PROMPT,
                input_variables=["query", "context", "sources"]
            )
            message = prompt.format(
                query=state.message,
                context=combined_context,
                sources=sources_text
            )

            # Generate the final response using the LLM
            response = await llm_structured.ainvoke(message)

            # Extract the content from the response and ensure it's a string
            if hasattr(response, "content"):
                final_answer = response.content if isinstance(

                    # type: ignore
                    response, FinalResponseOutput) else str(response.content)
            else:
                final_answer = str(response)

            logger.info(f"Generated final response: {final_answer}")
            state.final_answer = final_answer

        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            state.final_answer = "I apologize, but I encountered an error while generating your response. Please try asking your question again."

        if state.final_answer:
            state.short_term_memory.append(
                AIMessage(
                    content=state.final_answer,
                    additional_kwargs={
                        "sources": state.sources} if state.sources else {}
                )
            )
            # when adding AIMessage
            if state.user_id is not None:
                await save_conversation_memory(int(state.user_id), 'ai', state.final_answer)
            else:
                logger.error(
                    "state.user_id is None. Cannot save conversation memory.")

        return state
