CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/evaluation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=vitstr_base_patch16_224_withHeadSE --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=vitstr_base_patch16_224_withHeadSE_with_pretrained_weights


