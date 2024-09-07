#!/bin/bash

N=6  # Replace with the actual number of image/audio pairs

for i in $(seq 1 $N)
do
    echo "file 'aspect-scene-$i.png'" >> input.txt
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "p$i.mp3")
    echo "duration $duration" >> input.txt
done

for i in $(seq 1 $N)
do
    echo "file 'p$i.mp3'" >> audio_input.txt
done