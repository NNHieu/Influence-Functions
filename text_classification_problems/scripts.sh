# Training
python train.py datamodule=imdb datamodule.flip_percent=0 logger=wandb trainer=ddp trainer.devices=2
python train.py -m datamodule=imdb datamodule.flip_percent=0 logger=wandb trainer=ddp trainer.devices=\[2,3\] seed=121,122,123 trainer.max_epochs=10 test=False
python train.py -m datamodule=imdb datamodule.flip_percent=0 logger=wandb trainer=ddp trainer.devices=\[0,1\] seed=124,125 trainer.max_epochs=10 test=False
python train.py -m datamodule=snli datamodule.flip_percent=0.15,0.2 logger=wandb trainer=ddp trainer.devices=\[2,3\] seed=121 logger.wandb.log_model=False
python train.py -m datamodule=snli datamodule.flip_percent=0.05,0.1 logger=wandb trainer=ddp trainer.devices=\[0,1\] seed=121 logger.wandb.log_model=False

# Tracing
python run_tracing.py datamodule=snli tracer=gd train_output_dir=outputs/train/snli/flip0.2_bert/12345_2022-12-27_17-47-25 datamodule.train_batch_size=128

python run_tracing.py -m datamodule=imdb tracer=gd,if,tracin train_output_dir=
# Paths
outputs/imdb/flip0_bert/121_2023-01-02_12-11-48
outputs/imdb/flip0_bert/122_2023-01-02_12-11-48
outputs/imdb/flip0_bert/123_2023-01-02_12-11-48
outputs/imdb/flip0_bert/124_2023-01-02_12-12-57
outputs/imdb/flip0_bert/125_2023-01-02_12-12-57

#Paths
outputs/train/snli/multirun/flip0.05_bert/121_2022-12-31_20-19-06
outputs/train/snli/multirun/flip0.1_bert/121_2022-12-31_20-19-06
outputs/train/snli/multirun/flip0.15_bert/121_2022-12-31_20-22-31
outputs/train/snli/multirun/flip0.2_bert

# Tracing-Debug
python run_tracing.py datamodule=snli tracer=gd run_name=debug train_output_dir=outputs/train/snli_0.2_bert/2022-12-28_10-53-09 datamodule.train_batch_size=128

# Aggregation
python convert_result.py datamodule=imdb tracing_method=core.tracer.GradientBasedTracer,core.tracer.TracIn,core.tracer.IF train_output_dir=
