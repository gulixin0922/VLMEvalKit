# Qwen2.5-VL-7B
MASTER_PORT=63669 GPUS=8 srun -n1 --ntasks-per-node=1 --partition Intern5 --gres=gpu:8 --quotatype=reserved --job-name qwenvl_subject_eval --cpus-per-task=12 sh scripts/academic_subject_eval/evaluate.sh Qwen2.5-VL-7B  scripts/academic_subject_eval/qwen2_5_vl_7b_config.json

# Qwen2.5-VL-72B
MASTER_PORT=63668 GPUS=4 srun -n1 --ntasks-per-node=1 --partition Intern5 --gres=gpu:4 --quotatype=reserved --job-name qwenvl_subject_eval --cpus-per-task=12 sh scripts/academic_subject_eval/evaluate.sh Qwen2.5-VL-72B scripts/academic_subject_eval/qwen2_5_vl_72b_config.json --auto

# InternVL3-8B
MASTER_PORT=63667 GPUS=8 srun -n1 --ntasks-per-node=1 --partition Intern6 --gres=gpu:8 --quotatype=reserved --job-name internvl_subject_eval --cpus-per-task=12 sh scripts/academic_subject_eval/evaluate.sh InternVL3-8B scripts/academic_subject_eval/internvl3_8b_config.json
MASTER_PORT=63667 GPUS=8 srun -n1 --ntasks-per-node=1 --partition Intern6 --gres=gpu:8 --quotatype=reserved --job-name internvl_subject_eval --cpus-per-task=12 sh scripts/academic_subject_eval/evaluate.sh InternVL3-8B scripts/academic_subject_eval/internvl3_8b_config_0507.json
MASTER_PORT=63667 GPUS=8 srun -n1 --ntasks-per-node=1 --partition Intern6 --gres=gpu:8 --quotatype=reserved --job-name internvl_subject_eval --cpus-per-task=12 sh scripts/academic_subject_eval/evaluate.sh InternVL3-8B scripts/academic_subject_eval/internvl3_8b_config_0513.json

# InternVL3-78B
MASTER_PORT=63667 GPUS=4 srun -n1 --ntasks-per-node=1 --partition Intern5 --gres=gpu:4 --quotatype=reserved --job-name internvl_subject_eval --cpus-per-task=12 sh scripts/academic_subject_eval/evaluate.sh InternVL3-78B scripts/academic_subject_eval/internvl3_78b_config.json --auto

# chatgpt-4o-latest
srun -n1 --ntasks-per-node=1 --partition Intern5 --gres=gpu:0 --quotatype=reserved --job-name openai_subject_eval --cpus-per-task=10 sh scripts/academic_subject_eval/evaluate.sh chatgpt-4o-latest scripts/academic_subject_eval/openai_config.json

# claude-3-7-sonnet
srun -n1 --ntasks-per-node=1 --partition Intern5 --gres=gpu:0 --quotatype=reserved --job-name claude_subject_eval --cpus-per-task=10 sh scripts/academic_subject_eval/evaluate.sh claude-3-7-sonnet scripts/academic_subject_eval/claude_config.json