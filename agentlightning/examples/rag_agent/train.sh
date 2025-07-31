set -x

export OUTPUT_SYNC_DIR=./output
export OUTPUT_DIR="$OUTPUT_SYNC_DIR"
STDOUT_LOG_FILE="$OUTPUT_SYNC_DIR/stdout.log"
echo "Stdout and Stderr will be logged to: $STDOUT_LOG_FILE"

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

export N_GPUS=4
export BASE_MODEL=Qwen/Qwen3-1.7B
export DATA_DIR=data/musique
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=RAG_Qwen3-1.7B_grpo
export PROJECT_NAME=RAG_agent
export WANDB_RUN_ID=$(date +%Y%m%d_%H%M%S)

echo "Starting training script..."

TRAINER_EXIT_CODE=1 # Default to failure

# Retry logic parameters
quick_failure_threshold_seconds=600 # 10 minutes
max_consecutive_quick_failures=3
consecutive_quick_failures=0
attempt_count=0

while true; do
    attempt_count=$((attempt_count + 1))
    echo "----------------------------------------------------" | tee -a "$STDOUT_LOG_FILE"
    echo "Starting training attempt #$attempt_count..." | tee -a "$STDOUT_LOG_FILE"
    echo "Consecutive quick failures so far: $consecutive_quick_failures" | tee -a "$STDOUT_LOG_FILE"
    echo "----------------------------------------------------" | tee -a "$STDOUT_LOG_FILE"

    start_time=$(date +%s)

    stdbuf -oL python -m verl.trainer.main_ppo \
        agent_mode.enable=True \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.chat_scheduler=examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path=${BASE_MODEL} \
        data.train_files=${DATA_DIR}/musique_train.parquet \
        data.val_files=${DATA_DIR}/musique_dev_128.parquet \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
        trainer.n_gpus_per_node=${N_GPUS} \
        data.train_batch_size=32 \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        data.max_prompt_length=8192 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        trainer.val_before_train=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.000 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.3 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.default_local_dir=${OUTPUT_DIR} \
        trainer.nnodes=1 \
        trainer.save_freq=16 \
        trainer.test_freq=16 \
        trainer.total_epochs=2 $@ 2>&1 | tee -a "$STDOUT_LOG_FILE"

    current_attempt_exit_code=${PIPESTATUS[0]} # Get exit code of python, not tee
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "----------------------------------------------------" | tee -a "$STDOUT_LOG_FILE"
    echo "Training attempt #$attempt_count finished." | tee -a "$STDOUT_LOG_FILE"
    echo "Exit code: $current_attempt_exit_code" | tee -a "$STDOUT_LOG_FILE"
    echo "Duration: $duration seconds." | tee -a "$STDOUT_LOG_FILE"
    echo "----------------------------------------------------" | tee -a "$STDOUT_LOG_FILE"


    if [ "$current_attempt_exit_code" -eq 0 ]; then
        echo "Training successful on attempt #$attempt_count." | tee -a "$STDOUT_LOG_FILE"
        TRAINER_EXIT_CODE=0 # Set final exit code to success
        consecutive_quick_failures=0 # Reset counter on success
        break # Exit the loop
    else
        TRAINER_EXIT_CODE=$current_attempt_exit_code # Store the latest failure code
        echo "Training attempt #$attempt_count failed with exit code $current_attempt_exit_code." | tee -a "$STDOUT_LOG_FILE"

        if [ "$duration" -lt "$quick_failure_threshold_seconds" ]; then
            consecutive_quick_failures=$((consecutive_quick_failures + 1))
            echo "This was a quick failure (took $duration seconds)." | tee -a "$STDOUT_LOG_FILE"
            echo "Consecutive quick failures: $consecutive_quick_failures out of $max_consecutive_quick_failures." | tee -a "$STDOUT_LOG_FILE"
            if [ "$consecutive_quick_failures" -ge "$max_consecutive_quick_failures" ]; then
                echo "Maximum number of $max_consecutive_quick_failures consecutive quick failures reached. Stopping retries." | tee -a "$STDOUT_LOG_FILE"
                break # Exit the loop
            fi
        else
            echo "This was not a quick failure (took $duration seconds). Resetting consecutive quick failure count." | tee -a "$STDOUT_LOG_FILE"
            consecutive_quick_failures=0 # Reset counter if the failure was not quick (it was a long failure)
        fi
        echo "Retrying after a short delay..." | tee -a "$STDOUT_LOG_FILE"
        sleep 60
    fi
done

echo "----------------------------------------------------" | tee -a "$STDOUT_LOG_FILE"
if [ "$TRAINER_EXIT_CODE" -eq 0 ]; then
    echo "Training finished successfully." | tee -a "$STDOUT_LOG_FILE"
else
    echo "Training finished with failures after $attempt_count attempt(s)." | tee -a "$STDOUT_LOG_FILE"
fi
echo "Final trainer exit code: $TRAINER_EXIT_CODE" | tee -a "$STDOUT_LOG_FILE"
echo "----------------------------------------------------" | tee -a "$STDOUT_LOG_FILE"

echo "Attempting to upload $STDOUT_LOG_FILE to W&B..."
python new_scripts/yuge/upload_stdout_log.py \
    --project-name "$PROJECT_NAME" \
    --run-name "$EXPERIMENT_NAME" \
    --run-id "$WANDB_RUN_ID" \
    --log-file "$STDOUT_LOG_FILE"

exit $TRAINER_EXIT_CODE