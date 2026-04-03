#!/usr/bin/env bash
# Cut video at specified (start, end) timestamp ranges and concatenate into one output.
# Requires: ffmpeg
# Segments are re-encoded (libx264/aac) so timestamps are clean and playback doesn't
# buffer or glitch at cuts. Slower than stream-copy but avoids DTS/non-monotonous errors.
#
# Usage: ./cut_and_concat_video.sh INPUT_VIDEO OUTPUT_VIDEO
# Example: ./cut_and_concat_video.sh recording.mp4 highlights.mp4

set -euo pipefail

# -----------------------------------------------------------------------------
# Edit this list: each line is "START_SEC END_SEC" (in seconds). Clips are spliced in order.
# -----------------------------------------------------------------------------

# TIMESTAMPS=(
#   "0 52"
# ) # PEG

# TIMESTAMPS=(
#   "0 56"
# ) # LEG


# TIMESTAMPS=(
#   "0 36.5"
# ) # DRAWER


# TIMESTAMPS=(
#   "0 40.5"
# ) # rectnagle on wall

# TIMESTAMPS=(
#   "0 45"
# ) # cube stacking

TIMESTAMPS=(
  "0 43.5"
) # cupcake on plate

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_VIDEO OUTPUT_VIDEO" >&2
  echo "  Cuts INPUT_VIDEO at the (start,end) pairs in TIMESTAMPS and writes OUTPUT_VIDEO." >&2
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_VIDEO="$2"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Error: input video not found: $INPUT_VIDEO" >&2
  exit 1
fi

if [[ ${#TIMESTAMPS[@]} -eq 0 ]]; then
  echo "Error: TIMESTAMPS is empty. Add at least one 'START END' pair." >&2
  exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

CONCAT_LIST="$TMPDIR/concat.txt"
: > "$CONCAT_LIST"

for i in "${!TIMESTAMPS[@]}"; do
  read -r start end <<< "${TIMESTAMPS[$i]}"
  if [[ ! "$start" =~ ^[0-9]+\.?[0-9]*$ ]] || [[ ! "$end" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "Error: invalid timestamp pair at index $i: '${TIMESTAMPS[$i]}' (expected 'START END' numbers)" >&2
    exit 1
  fi
  duration=$(awk "BEGIN { printf \"%.3f\", $end - $start }")
  if [[ $(awk "BEGIN { print ($duration <= 0) ? 1 : 0 }") -eq 1 ]]; then
    echo "Error: invalid range at index $i: start=$start end=$end (end must be > start)" >&2
    exit 1
  fi
  seg="$TMPDIR/seg_${i}.mp4"
  echo "Extracting clip $((i+1))/${#TIMESTAMPS[@]}: ${start}s - ${end}s"
  # Re-encode so each segment has 0-based timestamps (avoids DTS out-of-order and buffering at cuts)
  # -map 0:a:0? makes audio optional (video-only input is fine)
  ffmpeg -y -loglevel warning -ss "$start" -i "$INPUT_VIDEO" -t "$duration" \
    -avoid_negative_ts make_zero -map 0:v:0 -map 0:a:0? \
    -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k \
    "$seg"
  printf "file '%s'\n" "$seg" >> "$CONCAT_LIST"
done

echo "Concatenating ${#TIMESTAMPS[@]} clips into $OUTPUT_VIDEO"
ffmpeg -y -loglevel warning -f concat -safe 0 -i "$CONCAT_LIST" -c copy "$OUTPUT_VIDEO"

echo "Done: $OUTPUT_VIDEO"
