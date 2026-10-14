#!/bin/bash

eta_upper_values="1.0 1.01"  # [1, 1.1, 1.2, ..., 2]
mu_values="-0.8"  # [-1.0, -0.9, -0.8, ..., -0.5]
discount_values="0.01"
v1_values="1.0"
pricing_multiply_values="1.0"

LOG_DIR="./logs"
mkdir -p $LOG_DIR

for eta_upper in $eta_upper_values; do
	for mu in $mu_values; do
		for discount in $discount_values; do
			for v1 in $v1_values; do
				v2=$(awk "BEGIN {print $v1 * 2.5}")
				for pricing_multiply in $pricing_multiply_values; do
					echo "Submitting job for eta_upper=$eta_upper, mu=$mu, discount=$discount, v1=$v1, v2=$v2, pricing_multiply=$pricing_multiply"
					LOG_FILE="$LOG_DIR/job_eta_${eta_upper}_mu_${mu}_discount_${discount}_v1_${v1}_v2_${v2}_pricing_multiply_${pricing_multiply}.log"
					sbatch --output="$LOG_FILE" submit.sh $eta_upper $mu $discount $v1 $v2 $pricing_multiply
				done
			done
		done
	done
done
