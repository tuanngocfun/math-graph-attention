from pytorch_lightning import Trainer

from data import MathGraphData
from model import GNNTrainer

if __name__ == "__main__":
    model = GNNTrainer(
        lr=1e-3,
    ).load_from_checkpoint(
        "checkpoint/egat_crohme/lightning_logs/version_15/checkpoints/epoch=21-val_acc=0.9542.ckpt"
    )

    dm = MathGraphData(batch_size=1, workers=1)

    trainer = Trainer(
        enable_checkpointing=False,
        fast_dev_run=False,
        deterministic=False,
        devices=[0],
    )

    if True:
        trainer.test(model, dm)
    else:
        trainer.fit(model, dm)
