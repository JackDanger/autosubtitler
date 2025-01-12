#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
usage:
    python autosubtitler.py --input_file <subtitles.srt|subtitles.ass>

description:
    This script parses an ASS or SRT subtitle file, focusing on the [Events] block
    in ASS or the entire content in SRT. It only attempts to translate the lines
    that appear as "Dialogue:" lines (in the case of ASS) or typical textual lines
    (in the case of SRT). The script then uses either Gemini or OpenAI, depending on
    which environment variable is set (GEMINI_API_KEY or OPENAI_API_TOKEN) to translate
    the "on-screen" text into English. A small context window of adjacent dialogue lines
    is provided to the LLM, but the LLM is strictly instructed to return only the
    translation of the target line (not the context). The final output is written to
    a file that has ".english" inserted before the original extension.

notes:
    - Requires either GEMINI_API_KEY or OPENAI_API_TOKEN environment variable.
      (If both are specified, Gemini takes precedence.)
    - Large files could lead to many LLM calls. Consider batching or rate-limiting.

Example:
    python autosubtitler.py --input_file example.ass
"""

import argparse
import logging
import os
import sys
import time
import traceback
from typing import List, Tuple, Optional

from openai import OpenAI  # Updated OpenAI client
from google import genai   # Gemini LLM

# Import the parser for ASS files:
from ass_parser import parse_ass_file, ASSFile, Dialogue


# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

MAX_RETRIES = 10
CONTEXT_WINDOW = 2  # Number of "dialogue" lines before/after the target line for context


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


def init_llm_clients(
    gemini_key: Optional[str],
    openai_token: Optional[str]
) -> Tuple[bool, Optional[genai.Client]]:
    """
    Initializes the LLM client(s) if available. Returns (use_gemini, gemini_client).
    Raises an error if required dependencies or environment variables are missing.
    """
    if not gemini_key and not openai_token:
        print_error_and_exit(
            "Neither GEMINI_API_KEY nor OPENAI_API_TOKEN is set. "
            "Please set one before running this script."
        )

    use_gemini = bool(gemini_key)

    # Check if user wants Gemini but it's not installed properly
    if use_gemini and genai is None:
        print_error_and_exit(
            "Gemini requires google-genai package. Install via `pip install google-genai`."
        )
    # Check if user wants OpenAI but no token
    if (not use_gemini) and (not openai_token):
        print_error_and_exit(
            "OpenAI requires OPENAI_API_TOKEN. Please set it or install the package."
        )

    gemini_client = None
    if use_gemini:
        gemini_client = genai.Client(api_key=gemini_key)

    return use_gemini, gemini_client


def infer(
    prompt: str,
    use_gemini: bool,
    gemini_client: Optional[genai.Client],
    gemini_model_name: str,
    openai_token: Optional[str],
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
                if not openai_token:
                    print_error_and_exit("OpenAI token not found.")
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
                    "after multiple attempts."
                )
    return ""


def translate_ass_dialogues(
    ass_file: ASSFile,
    use_gemini: bool,
    gemini_client,
    gemini_model_name: str,
    openai_token: Optional[str],
    openai_model_name: str,
) -> None:
    """
    Enumerate over the parsed Dialogue objects in the ASS file
    and translate only the 'text' portion. Then set dialogue.text
    to the translated version. This modifies 'ass_file' in-place.
    """
    for dialogue_obj in ass_file.dialogues:
        original_text = dialogue_obj.text  # everything after last comma
        if not original_text.strip():
            continue  # skip empty

        # Build a prompt specifically for the dialogue text
        prompt = (
            "You are an expert translator. Translate the following subtitle text to English:\n\n"
            f"{original_text}\n\n"
            "Return ONLY the translated text, no additional explanation."
        )
        translated_text = infer(
            prompt,
            use_gemini=use_gemini,
            gemini_client=gemini_client,
            gemini_model_name=gemini_model_name,
            openai_token=openai_token,
            openai_model_name=openai_model_name,
        )
        if translated_text.strip():
            dialogue_obj.text = translated_text

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

    # Determine extension
    valid_extensions = [".srt", ".ass"]
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext not in valid_extensions:
        print_error_and_exit(
            f"Unsupported file extension '{file_ext}'. Supported: {valid_extensions}"
        )

    # Load environment variables for Gemini / OpenAI
    gemini_key, openai_token = load_environment_vars()
    use_gemini, gemini_client = init_llm_clients(gemini_key, openai_token)

    # Choose model names
    gemini_model_name = "gemini-2.0-flash-exp"
    openai_model_name = "gpt-4o-mini"

    # Create output file path, e.g. "filename.ass" -> "filename.english.ass"
    base_name, extension = os.path.splitext(input_file)
    output_file = f"{base_name}.english{extension}"

    if file_ext == ".ass":
        # Use new ASS parser
        logging.info("Parsing ASS with ass_parser.py ...")
        with open(input_file, "r", encoding="utf-8") as f_in:
            all_lines = [line.rstrip("\r\n") for line in f_in]

        ass_file = parse_ass_file(all_lines)

        # Translate the dialogues in the ass_file
        logging.info("Translating dialogues in ASS file...")
        translate_ass_dialogues(
            ass_file,
            use_gemini=use_gemini,
            gemini_client=gemini_client,
            gemini_model_name=gemini_model_name,
            openai_token=openai_token,
            openai_model_name=openai_model_name,
        )

        # Reserialize the entire ASS
        logging.info("Re-serializing ASS file...")
        output_lines = ass_file.serialize()

        # Write the result
        with open(output_file, "w", encoding="utf-8") as f_out:
            for line in output_lines:
                f_out.write(line + "\n")

    else:  # file_ext == ".srt"
        print("NYI")
        pass

    logging.info(f"Translation complete. Output saved to: {output_file}")


if __name__ == "__main__":
    main()
