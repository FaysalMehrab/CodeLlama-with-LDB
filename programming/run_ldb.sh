
dataset="/home/mehrab-chowdhury/Desktop/LLMDebugger/input_data/humaneval/dataset/probs.jsonl"
model="codellama/CodeLlama-7b-Python-hf"
seedfile="/home/mehrab-chowdhury/Desktop/LLMDebugger/input_data/humaneval/seed/reflexion/seed.jsonl"  
output_dir="/home/mehrab-chowdhury/Desktop/LLMDebugger/output_data/ldb_HumanEval_Relexion2/Mbpp/$model/"
strategy="ldb"

python3 /home/mehrab-chowdhury/Desktop/LLMDebugger/programming/main.py \
  --run_name $output_dir \
  --root_dir "/home/mehrab-chowdhury/Desktop/LLMDebugger/output_data/$strategy/$(basename $dataset)/$model/" \
  --dataset_path $dataset \
  --strategy $strategy \
  --model $model \
  --seedfile $seedfile \
  --pass_at_k "1" \
  --max_iters "10" \
  --n_proc "1" \
  --port "8000" \
  --testfile "/home/mehrab-chowdhury/Desktop/LLMDebugger/input_data/humaneval/test/tests.jsonl" \
  --verbose


