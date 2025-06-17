from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data import MathGraphData
from model import GNNTrainer

if __name__ == "__main__":
    model = GNNTrainer(
        lr=1e-3,
    )
    # .load_from_checkpoint(
    #     "checkpoint/egat_crohme/lightning_logs/version_15/checkpoints/epoch=21-val_acc=0.9542.ckpt"
    # )

    dm = MathGraphData(batch_size=1, workers=1)

    trainer = Trainer(
        enable_checkpointing=True,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                filename="{epoch}-{val_acc:.4f}",
                save_top_k=5,
                monitor="val_seq_acc",
                mode="max",
            ),
        ],
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        default_root_dir="checkpoint/egat_crohme2019",
        deterministic=False,
        max_epochs=50,
        log_every_n_steps=50,
        devices=1,
        accelerator='gpu'
        # precision=16,
        # strategy='ddp',
    )

    if False:
        trainer.test(model, dm)
    else:
        trainer.fit(model, dm)
