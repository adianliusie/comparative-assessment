#!/bin/bash

# activate environment
source ~/.bashrc
conda activate torch2.0

# summeval:      consistency fluency coherency relevance
# topicalchat:   coherency naturalness continuity engagingness
# webnlg:        fluency grammar

# get flags
score='naturalness'
dataset='topicalchat'
device='cuda'
while getopts d:s:v: flag; do
    case "${flag}" in
        d) dataset=${OPTARG};;
        s) score=${OPTARG};;
        v) device=${OPTARG};;
    esac
done

echo $dataset $score $device

system='chatgpt'
# loop through systems
for i in 1 2; do
    python ../system_run.py --dataset $dataset --score-type $score --output-path output_text/$dataset/$system/$score/scoring-$i --system $system --prompt-id s$i --shuffle --device $device --max-len 2

    #python ../system_run.py --dataset $dataset --score-type $score --output-path output_text/$dataset/$system/$score/comparative-$i --system $system --prompt-id c$i --comparative --shuffle --device $device

done 
