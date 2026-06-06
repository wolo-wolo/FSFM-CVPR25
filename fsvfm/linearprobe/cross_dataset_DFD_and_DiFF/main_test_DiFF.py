# Re-export the existing clean ViT DiFF testing script so LP checkpoints can be
# evaluated with exactly the same protocol as simple fine-tuning.
import os
import sys
import runpy
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.abspath(os.path.join(HERE, '../../finetune/cross_dataset_DFD_and_DiFF'))
sys.path.insert(0, TARGET_DIR)
runpy.run_path(os.path.join(TARGET_DIR, 'main_test_DiFF.py'), run_name='__main__')
