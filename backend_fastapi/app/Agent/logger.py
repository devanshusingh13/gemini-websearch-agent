import logging
import os
import traceback
import sys

# Create a logger with a unique name for your Gemini Agent project
logger = logging.getLogger("gemini_agent")
logger.setLevel(logging.INFO)

# Ensure log directory exists
log_directory = "./logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# File Handler (logs saved to gemini_agent.log)
file_handler = logging.FileHandler("./logs/gemini_agent.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Console Handler (logs output to terminal)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Try to set console encoding to utf-8 if possible (Python 3.7+)
try:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
except Exception:
    pass  # If reconfigure fails or does not exist, ignore

# Define log message format with timestamp, level, file name, line number, and message
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent log messages from being propagated to the root logger (avoid duplicate logs)
logger.propagate = False

# Alias for easier import elsewhere
LOGGER = logger
