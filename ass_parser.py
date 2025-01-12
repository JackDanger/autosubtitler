#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
usage:
    python ass_parser.py --input_file <subtitle.ass> --output_file <subtitle.parsed.ass>

description:
    This script reads an ASS (Advanced SubStation Alpha) subtitle file,
    parses its sections:
      - [Script Info]
      - [V4+ Styles]
      - [Events]
    and stores them in data structures. It also parses 'Dialogue:' lines
    from the [Events] section into structured objects.

    After any potential processing (e.g., translation), the script can
    re-serialize the entire subtitle (including unchanged sections)
    and write it to an output file.

example:
    python ass_parser.py --input_file input.ass --output_file output.ass
"""

import argparse
import os
import sys
from typing import List, Dict, Optional


class ASSFile:
    """
    Represents a parsed ASS file, containing:
      - script_info: List[str]      (raw lines under [Script Info])
      - v4plus_styles: List[str]    (raw lines under [V4+ Styles])
      - events: List[str]           (raw lines under [Events], including Format: header)
      - dialogues: List[Dialogue]   (parsed from 'Dialogue:' lines in the [Events] section)
    """
    def __init__(self):
        self.script_info: List[str] = []
        self.v4plus_styles: List[str] = []
        self.events: List[str] = []
        self.dialogues: List["Dialogue"] = []

    def serialize(self) -> List[str]:
        """
        Reconstruct the entire ASS file from the stored data and return
        it as a list of lines (which can then be written to disk).
        """
        # Rebuild [Script Info] section
        output_lines = []
        if self.script_info:
            output_lines.append("[Script Info]")
            output_lines.extend(self.script_info)
            output_lines.append("")

        # Rebuild [V4+ Styles] section
        if self.v4plus_styles:
            output_lines.append("[V4+ Styles]")
            output_lines.extend(self.v4plus_styles)
            output_lines.append("")

        # Rebuild [Events] section
        if self.events:
            output_lines.append("[Events]")
            # The events section will have lines such as:
            #   Format: ...
            #   Dialogue: ...
            #   Dialogue: ...
            # We'll store the event lines in self.events,
            # but for lines beginning with 'Dialogue:',
            # we might want to rebuild from our Dialogue objects.

            # We'll create a map from original index to Dialogue object
            # so we can easily reconstruct them in the correct position.
            dialogue_lines_map = {
                d.original_index: d for d in self.dialogues
                if d.original_index is not None
            }

            # For each line in the events section, if it's a Dialogue,
            # we re-serialize from the Dialogue object; otherwise, keep it as is.
            for i, line in enumerate(self.events):
                # Check if this line was parsed as a Dialogue
                if i in dialogue_lines_map:
                    output_lines.append(dialogue_lines_map[i].serialize())
                else:
                    output_lines.append(line)
            output_lines.append("")

        return output_lines


class Dialogue:
    """
    Represents a single 'Dialogue:' line in ASS, parsed into fields:
      - layer
      - start
      - end
      - style
      - name
      - margin_l
      - margin_r
      - margin_v
      - effect
      - text
      - original_index: the index within the [Events] section so we can re-serialize in place
    """
    def __init__(
        self,
        layer: str,
        start: str,
        end: str,
        style: str,
        name: str,
        margin_l: str,
        margin_r: str,
        margin_v: str,
        effect: str,
        text: str,
        original_index: Optional[int] = None
    ):
        self.layer = layer
        self.start = start
        self.end = end
        self.style = style
        self.name = name
        self.margin_l = margin_l
        self.margin_r = margin_r
        self.margin_v = margin_v
        self.effect = effect
        self.text = text
        self.original_index = original_index

    @classmethod
    def from_line(cls, line: str, idx_in_events: int) -> Optional["Dialogue"]:
        """
        Attempt to parse a line of the form:
          Dialogue: 0,0:00:17.97,0:00:20.25,Default,Aiku,0000,0000,0000,,Gute Arbeit, du Genie.
        Returns a Dialogue object, or None if parsing fails.
        """
        # Strip leading/trailing
        line = line.strip()
        if not line.startswith("Dialogue:"):
            return None

        # Remove "Dialogue: " prefix
        # We want everything after "Dialogue: "
        parts_str = line[len("Dialogue:"):].strip()
        # parts_str should look like:
        # "0,0:00:17.97,0:00:20.25,Default,Aiku,0000,0000,0000,,Gute Arbeit, du Genie."
        parts = parts_str.split(",", 9)  # split into up to 10 parts
        # Typically the ASS format is 9 commas => 10 fields after "Dialogue: "
        #   1: layer
        #   2: start
        #   3: end
        #   4: style
        #   5: name
        #   6: margin_l
        #   7: margin_r
        #   8: margin_v
        #   9: effect
        #   10: text
        if len(parts) < 10:
            # Malformed dialogue line
            return None

        layer, start, end, style, name, margin_l, margin_r, margin_v, effect, text = parts
        return cls(
            layer=layer.strip(),
            start=start.strip(),
            end=end.strip(),
            style=style.strip(),
            name=name.strip(),
            margin_l=margin_l.strip(),
            margin_r=margin_r.strip(),
            margin_v=margin_v.strip(),
            effect=effect.strip(),
            text=text,  # keep raw text, might contain commas
            original_index=idx_in_events
        )

    def serialize(self) -> str:
        """
        Convert this Dialogue object back into a valid "Dialogue: ..." line.
        """
        # e.g.:
        # Dialogue: 0,0:00:17.97,0:00:20.25,Default,Aiku,0000,0000,0000,,Gute Arbeit, du Genie.
        line = (
            "Dialogue: "
            f"{self.layer},"
            f"{self.start},"
            f"{self.end},"
            f"{self.style},"
            f"{self.name},"
            f"{self.margin_l},"
            f"{self.margin_r},"
            f"{self.margin_v},"
            f"{self.effect},"
            f"{self.text}"
        )
        return line


def parse_ass_file(lines: List[str]) -> ASSFile:
    """
    Given all lines of an ASS file, separate them by sections:
      [Script Info], [V4+ Styles], [Events]
    Then parse out 'Dialogue:' lines into Dialogue objects for safe manipulation.
    """
    ass_file = ASSFile()
    current_section = None  # We'll store section name or None

    # We'll also keep track of the index of each line in the [Events] block
    # so that we can map them to Dialogue objects.
    events_index = 0

    for line in lines:
        trimmed = line.strip()

        # Detect section headers
        if trimmed.startswith("[") and trimmed.endswith("]"):
            current_section = trimmed  # e.g. "[Script Info]", "[V4+ Styles]", "[Events]"
            continue

        if current_section == "[Script Info]":
            ass_file.script_info.append(line)
        elif current_section == "[V4+ Styles]":
            ass_file.v4plus_styles.append(line)
        elif current_section == "[Events]":
            # We always store the line in events
            ass_file.events.append(line)

        else:
            # Lines outside recognized sections are basically unhandled or
            # part of "head" content.  Some .ass files have multiple or custom sections,
            # but let's keep it simple and do nothing with them, or append them
            # to script_info as a fallback.
            if current_section is None:
                # We might treat them as "script_info" lines
                ass_file.script_info.append(line)
            else:
                # Or store them in the nearest recognized section
                if current_section == "[Script Info]":
                    ass_file.script_info.append(line)
                elif current_section == "[V4+ Styles]":
                    ass_file.v4plus_styles.append(line)

    # Now parse dialogues from the [Events] lines
    # We do it after the main read so that we know each line's index
    for i, event_line in enumerate(ass_file.events):
        # Attempt to parse as Dialogue
        dialogue_obj = Dialogue.from_line(event_line, i)
        if dialogue_obj:
            ass_file.dialogues.append(dialogue_obj)

    return ass_file


def main():
    parser = argparse.ArgumentParser(description="ASS parser for deserialization and reserialization.")
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the input .ass file",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to the output .ass file",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: input file does not exist: {args.input_file}")
        sys.exit(1)

    with open(args.input_file, "r", encoding="utf-8") as f_in:
        lines = [l.rstrip("\r\n") for l in f_in]

    # Parse the ASS file
    ass_file = parse_ass_file(lines)

    # Example of enumerating dialogue lines:
    #   (Pretend we want to uppercase each dialogue's text for demonstration)
    for dialogue in ass_file.dialogues:
        # 'dialogue.text' is everything after the last comma in "Dialogue:" except we preserve any \N, etc.
        # We'll do a trivial example transformation: uppercase the text
        # You could call your LLM translator here instead.
        dialogue.text = dialogue.text.upper()

    # Now re-serialize
    output_lines = ass_file.serialize()
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for line in output_lines:
            f_out.write(line + "\n")

    print(f"Finished parsing and re-serializing: {args.output_file}")


if __name__ == "__main__":
    main()
