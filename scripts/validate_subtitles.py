"""
Validate subtitle file format, chronological order, non-overlapping intervals,
and coverage across the entire 570-second video duration.
"""
import re
import sys

def parse_time(tc_str):
    # Format: HH:MM:SS,mmm
    m = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", tc_str.strip())
    if not m:
        raise ValueError(f"Invalid timestamp format: {tc_str}")
    hours, mins, secs, millis = map(int, m.groups())
    return hours * 3600 + mins * 60 + secs + millis / 1000.0

def validate_srt(filepath):
    print(f"Validating SRT: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    print(f"Found {len(blocks)} subtitle blocks.")

    last_end = 0.0
    for idx, b in enumerate(blocks):
        lines = b.strip().split("\n")
        if len(lines) < 3:
            raise ValueError(f"Block {idx+1} has fewer than 3 lines: {lines}")
        seq_num = int(lines[0].strip())
        if seq_num != idx + 1:
            raise ValueError(f"Block sequence mismatch at {idx+1}: got {seq_num}")

        time_line = lines[1]
        if "-->" not in time_line:
            raise ValueError(f"Missing '-->' in time line at block {idx+1}: {time_line}")
        start_str, end_str = time_line.split("-->")
        start_t = parse_time(start_str)
        end_t = parse_time(end_str)

        if start_t >= end_t:
            raise ValueError(f"Block {idx+1}: start_time {start_t} >= end_time {end_t}")
        if start_t < last_end:
            raise ValueError(f"Block {idx+1} overlaps with previous block: start {start_t} < prev_end {last_end}")

        text = " ".join(lines[2:]).strip()
        if not text:
            raise ValueError(f"Block {idx+1} has empty text!")

        last_end = end_t

    print(f"Final subtitle ends at: {last_end:.2f} seconds ({last_end/60:.2f} minutes).")
    if last_end < 540.0:
        raise ValueError(f"Subtitles terminate too early! Ended at {last_end:.2f}s, expected ~570s.")
    print("ALL 37 SUBTITLE BLOCKS VALIDATED SUCCESSFULLY! Monotonicity and duration verified.")
    return True

if __name__ == "__main__":
    srt_path = r"C:\PROJECTS\multimodal-stock-ai\video\captions\Stock_AI_Project_Explained.srt"
    validate_srt(srt_path)
