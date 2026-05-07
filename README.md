## Machine Translation Exercise 3

### Repository Structure
- **configs/**: Contains three configuration files:
  - `deen_transformer_regular.yaml`: The baseline (Enc-pre, Dec-post).
  - `deen_transformer_pre.yaml`: Pure Pre-norm strategy.
  - `deen_transformer_post.yaml`: Pure Post-norm strategy.
- **scripts/**: 
  - `train.sh`: Main training script. To switch models, simply change the `model_name` variable inside the script.
  - `visualization.py`: A script to extract PPL data and generate charts.
- **models/**: I have preserved `validations.txt` and `*.hyps` for each model as evidence of training and translation quality. (Large `.ckpt` files are excluded).
