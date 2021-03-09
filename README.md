# LSMA_HW1
Train : 
python2 train_mlp.py ~/SoundNet-tensorflow/output_10to17/ ./bof_K75 250 labels/train_split.csv labels/val_split.csv models/mfcc-sn_fc.mlp.model

Testing:
python2 test_mlp.py  ~/SoundNet-tensorflow/output_10to17  bof_K75/ 250  labels/test_for_student.label models/mfcc-sn_fc.mlp.model mfcc-sn7to14.mlp.csv
