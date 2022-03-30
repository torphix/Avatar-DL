1. conda env asr
2. used as part of the data processing pipeline to annotate audio
<!-- Command to run -->

python3 train.py data.train_path=/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/stt/data/jocko.json trainer.gpus=1 data.val_path=/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/stt/data/jocko.json data.batch_size=8 checkpoint.filepath=/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/stt/checkpoints/lightning_logs/version_2/checkpoints/epoch.365.step.2561.ckpt trainer.default_root_dir=/home/j/Desktop/Programming/AI/DeepLearning/la_solitudine/stt/checkpoints 