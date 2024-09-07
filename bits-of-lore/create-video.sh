ffmpeg -f concat -safe 0 -i input.txt -f concat -safe 0 -i audio_input.txt \
  -filter_complex "[1:a]apad[a]" -map 0:v -map "[a]" \
  -c:v libx264 -c:a aac -shortest output.mp4