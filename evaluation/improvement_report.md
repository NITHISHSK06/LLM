# Model Improvement Report

No V2 model comparison has been run yet. The current raw instruction dataset contains one valid example, which is insufficient for a separate training and validation split without data leakage. Add more independent instruction examples, run `python training/prepare_instruction_data.py`, train V2 from `checkpoints/best_model.pt`, and then run `python evaluation/compare_models.py`.

No improvement claim is made.
