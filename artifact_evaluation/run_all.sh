
MODEL=meta-llama/Llama-3.1-8B-Instruct

mkdir results

# Start MPS
bash ../scripts/start_mps.sh

# Launch server
nohup python -m sglang.launch_server --model-path ${MODEL} --disable-radix-cache --load-format dummy --enable-bullet-engine > results/bullet.log 2>&1 &

while ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30000/health | grep -q 200; do 
    echo "Waiting for server to start..."
    sleep 10
done
sleep 5

# Run benchmark

for rate in 10 12.5 15 17.5 20 22.5 25
do
    echo "Running benchmark with request rate: $rate"

    python ../python/sglang/bench_serving.py \
            --backend sglang \
            --dataset-name sharegpt \
            --num-prompts 1000 \
            --host 127.0.0.1 \
            --port 30000 \
            --model Meta-Llama-3.1-8B-Instruct \
            --request-rate $rate \
            --output-file results/result.json > results/benchmark.log 2>&1
done

# Kill MPS
bash ../scripts/killall_sglang.sh
bash ../scripts/kill_mps.sh


# Launch server
nohup python -m sglang.launch_server --model-path ${MODEL} --disable-radix-cache --load-format dummy --chunked-prefill-size 1024 > results/sglang.log 2>&1 &

while ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30000/health | grep -q 200; do 
    echo "Waiting for server to start..."
    sleep 10
done
sleep 5

# Run benchmark

for rate in 10 12.5 15 17.5 20 22.5 25
do
    echo "Running benchmark with request rate: $rate"

    python ../python/sglang/bench_serving.py \
            --backend sglang \
            --dataset-name sharegpt \
            --num-prompts 1000 \
            --host 127.0.0.1 \
            --port 30000 \
            --model Meta-Llama-3.1-8B-Instruct \
            --request-rate $rate \
            --output-file results/result.json > results/benchmark.log 2>&1
done

bash ../scripts/killall_sglang.sh

# Plot the figures
python ./plot.py
