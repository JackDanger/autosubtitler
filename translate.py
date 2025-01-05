#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
usage:
    python autosubtitler.py --input_file <subtitles.srt|subtitles.ass>

description:
    This script reads a subtitle file (SRT or ASS) and attempts to translate
    it into English by calling an LLM (Gemini or OpenAI). The translated
    subtitles are written to a new file with '.english' inserted before
    the original file extension.

notes:
    - Requires either GEMINI_API_KEY or OPENAI_API_TOKEN environment variable.
    - If both are specified, Gemini takes precedence.
    - This script only demonstrates a basic approach to subtitle translation
      via large language models. For production or large subtitles, you may
      want to chunk your calls to avoid token size limitations.
"""

import argparse
import logging
import os
import re
import sys
import time
import traceback
from typing import Optional

# If you have the respective libraries installed:
# pip install google-genai openai
try:
    import openai
except ImportError:
    openai = None

try:
    from google import genai
except ImportError:
    genai = None

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def print_error_and_exit(message: str) -> None:
    """Prints an error message and exits the script."""
    print(f"Error: {message}")
    sys.exit(1)

# Read environment variables for Gemini / OpenAI
gemini_key = os.environ.get("GEMINI_API_KEY")
openai_token = os.environ.get("OPENAI_API_TOKEN")

# Default model names, if needed
gemini_model_name = "gemini-2.0-flash-exp"
openai_model_name = "gpt-4o-mini"

if not gemini_key and not openai_token:
    print_error_and_exit(
        "Neither GEMINI_API_KEY nor OPENAI_API_TOKEN is set. "
        "Please set one before running this script."
    )

use_gemini = bool(gemini_key)

if use_gemini and genai is None:
    print_error_and_exit(
        "Gemini requires google-genai package. Install via `pip install google-genai`."
    )
elif (not use_gemini) and openai is None:
    print_error_and_exit(
        "OpenAI requires openai package. Install via `pip install openai`."
    )

# Initialize the Gemini client if using Gemini
if use_gemini:
    gemini_client = genai.Client(api_key=gemini_key)

MAX_RETRIES = 10

def infer(prompt: str) -> str:
    """
    Sends a prompt to either Gemini or OpenAI, returning the text response.
    Retries up to MAX_RETRIES times if a rate limit or similar error occurs.
    """
    backend_name = "Gemini" if use_gemini else "OpenAI"
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            logging.debug(
                f"[DEBUG] Attempt {attempt}/{MAX_RETRIES} to call {backend_name} API."
            )
        try:
            if use_gemini:
                response = gemini_client.models.generate_content(
                    model=gemini_model_name,
                    contents=prompt.strip(),
                )
                return response.text.strip()
            else:
                openai.api_key = openai_token
                completion = openai.ChatCompletion.create(
                    model=openai_model_name,
                    messages=[{"role": "user", "content": prompt.strip()}],
                    temperature=1.0,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return completion.choices[0].message.content.strip()

        except Exception as e:
            err_str = str(e)
            logging.debug(
                f"[DEBUG] {backend_name} API call attempt {attempt} failed: {err_str}"
            )
            traceback.print_exc()
            # If rate-limit, back off and retry
            if "rate limit" in err_str.lower() or "429" in err_str.lower():
                wait_time = 2 ** (attempt - 1)
                logging.debug(
                    "[DEBUG] Rate limit encountered. Waiting %s seconds before retry...",
                    wait_time,
                )
                time.sleep(wait_time)
            if attempt == MAX_RETRIES:
                print_error_and_exit(
                    f"Failed to get a valid response from {backend_name} after multiple attempts."
                )
    return ""

def main():
    parser = argparse.ArgumentParser(description="Auto-subtitle translator.")
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the .srt or .ass file to be translated.",
    )
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.isfile(input_file):
        print_error_and_exit(f"Input file does not exist: {input_file}")

    # Ensure file extension is either .srt or .ass
    valid_extensions = [".srt", ".ass"]
    ext = os.path.splitext(input_file)[1].lower()
    if ext not in valid_extensions:
        print_error_and_exit(
            f"Invalid file extension '{ext}'. Supported: {valid_extensions}"
        )

    # Read the original subtitle file
    try:
        with open(input_file, "r", encoding="utf-8") as f_in:
            original_subtitles = f_in.read()
    except Exception as e:
        print_error_and_exit(f"Error reading input file: {e}")

    # Simple prompt to translate the entire subtitle content to English
    prompt = (
        "You are an expert at translating subtitles into English. "
        "Please translate the following subtitle content into English, "
        "maintaining the original timing/formatting lines (numbers, timestamps, etc.), "
        "but replace all non-English dialogue with English:\n\n"
        f"{original_subtitles}"
    )

    logging.info("Sending prompt to LLM for translation...")
    translated_subtitles = infer(prompt)

    # Create the output file path, e.g. "filename.srt" -> "filename.english.srt"
    base_name, extension = os.path.splitext(input_file)
    output_file = base_name + ".english" + extension

    try:
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.write(translated_subtitles)
        logging.info(f"Translation complete. Output saved to: {output_file}")
    except Exception as e:
        print_error_and_exit(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()