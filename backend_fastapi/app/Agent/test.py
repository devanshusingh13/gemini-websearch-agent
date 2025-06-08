import asyncio
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from ..database.database_operations import init_models
from .Agentmain import Agent
from .Agent_state import GeminiAgentState
from .logger import logger


def print_conversation_history(history):
    """Print formatted conversation history with message types and content."""
    if not history:
        print("\n💬 No conversation history available.")
        return

    print("\n=== Conversation History ===")
    for i, msg in enumerate(history):
        try:
            # Get message content safely
            content = str(msg.content) if hasattr(msg, 'content') else str(msg)

            # Truncate long messages
            if len(content) > 100:
                content = content[:97] + "..."

            # Get message type name
            msg_type = type(msg).__name__

            # Print formatted message
            print(f"[{i}] {msg_type}: {content}")

        except Exception as e:
            logger.error(f"Error printing message {i}: {e}")
            continue

    print("=== End of Conversation History ===\n")


async def interactive_conversation():
    """Interactive conversation loop with memory management."""
    agent = Agent()
    user_id = 1  # fixed test user
    conversation_history = []  # Start with empty history - agent will handle memory

    async def handle_special_commands(user_input: str) -> bool:
        """Handle special testing and debugging commands."""
        commands = {
            "recall": handle_recall,
            "memory": handle_memory_status,
            "clear": handle_clear_memory,
            "help": handle_help
        }
        cmd = user_input.lower()
        if cmd in commands:
            await commands[cmd]()
            return True
        return False

    async def handle_recall():
        """Show relevant memories for current context."""
        if state and state.message:
            memories, summaries = await agent.get_conversation_context(user_id, state.message)
            print("\n📜 Relevant Memories:")
            for m in memories:
                print(f"Memory: {m}")
            print("\n📝 Relevant Summaries:")
            for s in summaries:
                print(f"Summary: {s}")
        else:
            print("\n⚠️ No current context to recall memories for")

    async def handle_memory_status():
        """Show current memory status."""
        print(f"\n🧠 Current Memory Status:")
        print(f"Short-term memory size: {len(conversation_history)}")
        if conversation_history:
            print("\nRecent interactions:")
            print_conversation_history(conversation_history[-3:])

    async def handle_clear_memory():
        """Clear current conversation history."""
        conversation_history.clear()
        print("\n🧹 Conversation history cleared")

    async def handle_help():
        """Show available commands."""
        print("\n📌 Available Commands:")
        print("- recall: Show relevant memories")
        print("- memory: Show current memory status")
        print("- clear: Clear current conversation")
        print("- help: Show this help message")
        print("- exit/quit: End conversation")

    # Main conversation loop
    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting conversation.")
            break

        if not user_input:  # Skip empty inputs
            continue

        if await handle_special_commands(user_input):
            continue

        try:
            # Create state with minimal context
            state = GeminiAgentState(
                message=user_input,
                user_id=user_id,
                short_term_memory=conversation_history
            )

            # Get agent response
            result_state = await agent.graph.ainvoke(state)

            # Display response
            response = result_state.get(
                "response") or result_state.get("final_answer")
            if response:
                print(f"\n{response}")
                # Update conversation history
                conversation_history.append(HumanMessage(content=user_input))
                conversation_history.append(AIMessage(content=response))
            else:
                print("\nSorry, I don't have an answer.")

        except Exception as e:
            logger.error(f"Error during conversation: {e}")
            print(f"\n❌ Error: {str(e)}")
            continue


async def main():
    """Initialize database and start conversation."""
    try:
        logger.info("🔧 Initializing database...")
        await init_models()
        logger.info("✅ Database initialized.")
        print_conversation_history([])
        await interactive_conversation()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print("❌ Failed to start conversation")


if __name__ == "__main__":
    asyncio.run(main())
