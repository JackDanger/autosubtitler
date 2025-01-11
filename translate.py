#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
usage:
    python autosubtitler.py --input_file <subtitles.srt|subtitles.ass>

description:
    This script reads a subtitle file (SRT or ASS) line by line with resilience
    to potentially malformed inputs. It translates each line individually into
    English using an LLM (Gemini or OpenAI), providing a small context window
    (previous and next lines) for better translation accuracy. The script
    ensures that only the target line is returned from the LLM, not the entire
    context. The translated file is written with the same base name but inserts
    '.english' before the extension (e.g., "video.srt" -> "video.english.srt").

notes:
    - Requires either GEMINI_API_KEY or OPENAI_API_TOKEN environment variable.
      (If both are specified, Gemini takes precedence.)
    - For large files, this line-by-line approach can be expensive and may
      require chunking or other optimizations.

Example:
    python autosubtitler.py --input_file example.srt
"""

import argparse
import logging
import os
import sys
import time
import traceback
from typing import List, Tuple, Optional
from openai import OpenAI
from google import genai

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

MAX_RETRIES = 10
CONTEXT_WINDOW = 2  # Number of lines before and after to provide as context


def print_error_and_exit(message: str) -> None:
    """
    Prints an error message and exits the script.
    """
    print(f"Error: {message}")
    sys.exit(1)


def load_environment_vars() -> Tuple[Optional[str], Optional[str]]:
    """
    Loads Gemini or OpenAI tokens from environment variables.
    Returns a tuple of (gemini_key, openai_token).
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_token = os.environ.get("OPENAI_API_TOKEN")
    return gemini_key, openai_token


def init_llm_clients(gemini_key: Optional[str], openai_token: Optional[str]):
    """
    Initializes the LLM client(s) if available. Returns (use_gemini, gemini_client).
    Raises an error if required dependencies are missing.
    """
    if not gemini_key and not openai_token:
        print_error_and_exit(
            "Neither GEMINI_API_KEY nor OPENAI_API_TOKEN is set. "
            "Please set one before running this script."
        )

    use_gemini = bool(gemini_key)

    # Check if user wants Gemini but it's not installed
    if use_gemini and genai is None:
        print_error_and_exit(
            "Gemini requires google-genai package. Install via `pip install google-genai`."
        )
    # Check if user wants OpenAI but it's not installed
    if (not use_gemini) and openai_token is None:
        print_error_and_exit(
            "OpenAI requires openai package. Install via `pip install openai`."
        )

    gemini_client = None
    if use_gemini:
        gemini_client = genai.Client(api_key=gemini_key)

    return use_gemini, gemini_client


def infer(
    prompt: str,
    use_gemini: bool,
    gemini_client,
    gemini_model_name: str,
    openai_token: str,
    openai_model_name: str,
) -> str:
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
            if use_gemini and gemini_client:
                response = gemini_client.models.generate_content(
                    model=gemini_model_name,
                    contents=prompt.strip(),
                )
                return response.text.strip()
            else:
                client = OpenAI(api_key=openai_token)
                completion = client.chat.completions.create(
                    model=openai_model_name,
                    messages=[{"role": "user", "content": prompt.strip()}],
                    response_format={"type": "text"},
                    temperature=1,
                    max_completion_tokens=2048,
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
                    f"Failed to get a valid response from {backend_name} "
                    f"after multiple attempts."
                )
    return ""


def parse_subtitle_file(file_path: str) -> List[str]:
    """
    Reads the file line by line and returns a list of lines.
    Implements basic resilience to malformed inputs by stripping
    trailing spaces and ignoring leading/trailing blank lines.
    """
    lines = []
    try:
        with open(file_path, "r", encoding="utf-8") as file_in:
            for raw_line in file_in:
                line = raw_line.rstrip("\r\n")
                # Accept the line as-is (including empty lines),
                # but strip trailing spaces to be consistent
                lines.append(line.strip(" \t"))
    except Exception as e:
        print_error_and_exit(f"Error reading input file: {e}")

    # Optionally remove leading/trailing blank lines
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def build_line_prompt(
    lines: List[str], index: int, window: int, line_translation_only: bool = True
) -> str:
    """
    Constructs a prompt for the LLM, providing a few lines before and after
    the target line for context. The prompt instructs the LLM to ONLY translate
    the target line (i.e., lines[index]) and not re-translate or alter context lines.

    :param lines: List of lines from the subtitle file.
    :param index: Current line index to be translated.
    :param window: Number of lines before/after as context.
    :param line_translation_only: If True, prompt enforces returning ONLY the
                                  translated target line.
    :return: Constructed prompt string.
    """
    start_ctx = max(0, index - window)
    end_ctx = min(len(lines), index + window + 1)

    # Provide the lines in context
    context_lines = lines[start_ctx:end_ctx]

    # Because we might only want to translate lines that are actual text,
    # and keep timing or format lines as is, we instruct the LLM to produce
    # output ONLY for the 'target' line. We'll identify that line with a marker.
    target_line = lines[index]

    prompt_intro = (
        "You are an expert translator for subtitles. Below is a block of subtitle text. "
        "Some lines are provided as context (do not translate them). "
        "One line is marked as the TARGET line: only translate that line into English. "
        "Do not translate or alter context lines. Return ONLY the translated text, "
        "with no additional formatting, explanations, or mention of the context.\n\n"
    )
    formatted_context = []
    for i, text_line in enumerate(context_lines, start=start_ctx):
        if i == index:
            # Mark the target line
            formatted_context.append(f"[TARGET] {text_line}")
        else:
            formatted_context.append(f"[CONTEXT] {text_line}")

    # We'll assemble the prompt. We specifically instruct the LLM to
    # return only the translation for [TARGET].
    prompt = prompt_intro + "\n".join(formatted_context)

    if line_translation_only:
        prompt += (
            "\n\nRemember: Return only the translated version of the [TARGET] line.\n"
        )

    return prompt


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

    valid_extensions = [".srt", ".ass"]
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext not in valid_extensions:
        print_error_and_exit(
            f"Invalid file extension '{file_ext}'. Supported: {valid_extensions}"
        )

    gemini_key, openai_token = load_environment_vars()
    use_gemini, gemini_client = init_llm_clients(gemini_key, openai_token)

    # Model names (customize if desired)
    gemini_model_name = "gemini-2.0-flash-exp"
    openai_model_name = "gpt-4o-mini"

    # Parse the original subtitle file lines
    logging.info("Reading and parsing subtitle file...")
    lines = parse_subtitle_file(input_file)

    # Prepare output lines
    translated_lines = []

    # For each line, if it's likely dialogue/text, we attempt translation;
    # if it looks like a timestamp or styling, we keep it as is.
    # In a real scenario, you might use heuristics or regex to detect
    # which lines are textual vs. timing info.
    for i, line in enumerate(lines):
        # Simple check to skip lines that match SRT index or time range
        # or ASS section. This is naive, but can help avoid unneeded calls:
        if (
            line.isdigit()
            or "-->" in line
            or "[Events]" in line
            or "Format:" in line
            or "Dialogue:" in line
            or not line
        ):
            translated_lines.append(line)
            continue

        # Build prompt with context
        prompt = build_line_prompt(lines, i, CONTEXT_WINDOW)

        # Translate only the target line
        logging.debug(f"Translating line {i} with context window {CONTEXT_WINDOW}...")
        translated_line = infer(
            prompt,
            use_gemini=use_gemini,
            gemini_client=gemini_client,
            gemini_model_name=gemini_model_name,
            openai_token=openai_token,
            openai_model_name=openai_model_name,
        )
        print(translated_line)

        # If there's no result or an error, fallback to the original line
        translated_line = translated_line if translated_line else line
        translated_lines.append(translated_line)

    # Create the output file path, e.g. "filename.srt" -> "filename.english.srt"
    base_name, extension = os.path.splitext(input_file)
    output_file = f"{base_name}.english{extension}"

    # Write out the translated lines
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            for t_line in translated_lines:
                out_f.write(t_line + "\n")
        logging.info("Translation complete. Output saved to: %s", output_file)
    except Exception as e:
        print_error_and_exit(f"Error writing output file: {e}")


if __name__ == "__main__":
    main()
