import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # barra avanzamento training

from CHALLENGE.dataset import DepthDataset
from CHALLENGE.model import DepthEstimationModel
from CHALLENGE.utils import visualize_img, ssim


class EarlyStopping:

    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset se la loss migliora
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # Stop training se la loss non migliora


class Solver:

    def __init__(self, args):

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.args.is_train:  # per il training
            self.train_data = DepthDataset(train=DepthDataset.TRAIN, data_dir=args.data_dir, transform=None)
            self.val_data = DepthDataset(train=DepthDataset.VAL, data_dir=args.data_dir, transform=None)

            # serve per barra avanzamento training
            self.train_loader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True,
                                           num_workers=4)

            self.net = DepthEstimationModel().to(self.device)

            # Ottimizzatore e funzione di loss e scheduler LR

            self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-5)
            # weight-decay aggiunto dopo vedi su notepad
            self.criterion = lambda output, depth: 0.6 * F.mse_loss(output, depth) + 0.4 * (1 - ssim(output, depth))
            # iniziato con 0,7*mse e 0,3*ssim
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-6)

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)

            # Early Stopping
            self.early_stopping = EarlyStopping(patience=10, delta=0.001)

            # Ripresa TRAINING da checkpoint
            self.start_epoch = 0  # Default partenza da epoca 0
            ckpt_path = os.path.join(self.args.ckpt_dir, self.args.ckpt_file)

            if os.path.exists(ckpt_path):
                print(f" Training ripreso  dal checkpoint: {ckpt_path}")
                self.net.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
                self.start_epoch = int(
                    self.args.ckpt_file.split("_")[-1].split(".")[0]) + 1  # Estrazione n° epoca dal nome

        else:  # per il test
            # prova con .VAL e poi rimetti .TEST prima di inviare
            self.test_set = DepthDataset(train=DepthDataset.TEST, data_dir=self.args.data_dir)
            self.net = DepthEstimationModel().to(self.device)  # aggiunto io rispetto al template
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)

            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True))

    def fit(self):
        for epoch in range(self.start_epoch, self.args.max_epochs):
            self.net.train()
            total_train_loss = 0.0

            print(f"\n Epoca {epoch + 1}/{self.args.max_epochs} - Training in corso...")

            # Barra di avanzamento delle batch
            progress_bar = tqdm(self.train_loader, desc=f"  Epoch {epoch + 1}", leave=False)

            for i, (images, depth) in enumerate(progress_bar):
                images, depth = images.to(self.device), depth.to(self.device)

                self.optimizer.zero_grad()
                output = self.net(images)
                loss = self.criterion(output, depth)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

                # Vis Loss media nella barra
                progress_bar.set_postfix({"Batch Loss": loss.item()})

                # Chiama Visualize img
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(), depth[0].cpu(), output[0].cpu().detach(),
                                  suffix=f"Epoch {epoch}, Batch {i}")

            avg_train_loss = total_train_loss / len(self.train_loader)

            val_loss = self.evaluate(DepthDataset.VAL)

            print(f" Epoca {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Controllo LR
            self.scheduler.step(val_loss)
            print(f" Learning Rate attuale: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Controllo Early Stopping
            self.early_stopping.step(val_loss)
            if self.early_stopping.early_stop:
                print(" Early Stopping attivato - Fine training")
                break

            # Salvataggio checkpoint
            self.save(self.args.ckpt_dir, self.args.ckpt_name, epoch)

    def evaluate(self, sett):

        args = self.args
        if sett == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "TRAIN"
        elif sett == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        # loader = self.train_loader if set == DepthDataset.TRAIN else self.val_loader

        self.net.eval()

        rmse_acc = 0.0
        ssim_acc = 0.0
        total_loss = 0.0

        print("\n Evaluation...")

        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                images, depth = images.to(self.device), depth.to(self.device)
                output = self.net(images)

                loss = self.criterion(output, depth)
                total_loss += loss.item()

                rmse_acc += torch.sqrt(F.mse_loss(output, depth)).item()
                ssim_acc += ssim(output, depth).item()

                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(), depth[0].cpu(), output[0].cpu().detach(), suffix=suffix)

        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader))

        avg_loss = total_loss / len(loader)
        print(f" Val Loss: {avg_loss:.4f}")
        return avg_loss

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
        print(f" Checkpoint salvato: {save_path}")

    def test(self):

        # aggiunto io rispetto al template (per evitare problemi con BatchNorm/Dropout/ecc) -  ###
        self.net.eval()

        loader = DataLoader(self.test_set,  # prova e poi rimetti self.test_set
                            batch_size=self.args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="TEST")
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))
