dataset="/home/mehrab-chowdhury/Desktop/LLMDebugger/input_data/humaneval/dataset/probs.jsonl"
model="codellama/CodeLlama-7b-Python-hf"  
output_dir="/home/mehrab-chowdhury/Desktop/LLMDebugger/output_data/simple/dataset"
strategy="simple"

python /home/mehrab-chowdhury/Desktop/LLMDebugger/programming/main.py \
  --run_name "my_run" \
  --root_dir "/home/mehrab-chowdhury/Desktop/LLMDebugger/output_data/$strategy/$(basename $dataset)/$model/" \
  --dataset_path "$dataset" \
  --strategy "$strategy" \
  --model "$model" \
  --n_proc "1" \
  --testfile "/home/mehrab-chowdhury/Desktop/LLMDebugger/input_data/humaneval/test/tests.jsonl" \
  --verbose \
  --port "8000"
