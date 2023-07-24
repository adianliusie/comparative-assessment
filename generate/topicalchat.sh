#!/bin/bash

score='naturalness'
device='cuda:0'
dataset='topicalchat'

# coherency naturalness continuity engagingness
for system in flant5-base flant5-large flant5-xl flant5-xxl; do
    for i in 1 2 3 4; do
	python system_run.py --dataset $dataset --output-path output_text/summeval/$system/$score/scoring-$i --system $system --prompt-id s$i --shuffle --device $device

	python system_run.py --dataset $dataset --output-path output_text/summeval/$system/$score/comp-probs-$i --system $system --probs --prompt-id c$i --comparative --shuffle --device $device

        # python system_run.py --dataset $dataset --output-path output_text/summeval/$system/$score/comp-$i --system $system --prompt-id c$i --comparative --shuffle --device $device

    done;
done 
