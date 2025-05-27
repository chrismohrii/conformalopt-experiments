#!/bin/bash


# Run the final command separately
# python3 main.py "elec" & # Electricity dataset large: commented out for efficiency. 
python3 preregistered.py & 

stocks=("MSFT" "AMZN" "GOOGL" "daily-climate")
models=("ar" "theta" "prophet" "transformer")

for stock in "${stocks[@]}"
do
    for model in "${models[@]}"
    do
        python3 main.py "$stock" "$model" & 
    done
done

