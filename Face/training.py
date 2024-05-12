from datasets.dataset_imdb import SiameseIMDBDataset as SiameseDataset
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import model


sets = SiameseDataset("/media/johannes/data2/data/imdb")

train_size = int(0.8 * len(sets))
test_size = len(sets) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(sets, [train_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=7, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=7)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min")
acc = 'gpu' if torch.cuda.is_available() else 'cpu'
trainer = pl.Trainer(accelerator=acc, devices=1, max_epochs=50,
                        accumulate_grad_batches=64,
                        callbacks=[early_stop_callback], detect_anomaly=False)

m = model.FaceModel()

# Train the model
trainer.fit(m, train_loader, val_dataloaders=test_loader)

torch.save(m.state_dict(), "./weights/triplet.pt")