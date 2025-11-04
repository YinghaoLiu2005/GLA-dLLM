# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_path=/data/yinghaoliu/Fast-dLLM/models/finetune_fast_dLLM_v2_1.5B/checkpoint-4000

# build output directory
output_dir="evaluation_results"
mkdir -p ${output_dir}

echo "Starting GSM8K evaluation..."
task=gsm8k
accelerate launch eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True > ${output_dir}/gsm8k_results.log 2>&1
echo "GSM8K evaluation finished. Results saved to ${output_dir}/gsm8k_results.log"


echo "Starting Minerva Math evaluation..."
task=minerva_math
accelerate launch eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True > ${output_dir}/minerva_math_results.log 2>&1
echo "Minerva Math evaluation finished. Results saved to ${output_dir}/minerva_math_results.log"


echo "Starting IFEval evaluation..."
task=ifeval
accelerate launch eval.py --tasks ${task} --batch_size 32 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True > ${output_dir}/ifeval_results.log 2>&1
echo "IFEval evaluation finished. Results saved to ${output_dir}/ifeval_results.log"
