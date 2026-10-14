#!/bin/bash

eta_upper_values="0.76 0.77 0.78 0.79"  
mu_values="-0.3 -0.4 -0.5 -0.6" 
discount_values="0.01"

LOG_DIR="./logs"
mkdir -p $LOG_DIR

for eta_upper in $eta_upper_values; do
	for mu in $mu_values; do
		for discount in $discount_values; do
			echo "Submitting job for eta_upper=$eta_upper, mu=$mu, discount=$discount"
           		sbatch submit_10dim.sh $eta_upper $mu $discount 
		done
	done
done
