from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torchvision.models import resnet18
import torch.nn as nn
import torch
import logging
from tqdm import tqdm

class VolClsModel(nn.Module):
    def __init__(self, args):
        super(VolClsModel, self).__init__()
        self.args = args
        self.model = resnet18(weights="IMAGENET1K_V1" if args.pretrained else None)
        outchannels, kernelsize, stride, padding = self.model.conv1.out_channels, self.model.conv1.kernel_size, self.model.conv1.stride, self.model.conv1.padding
        self.model.conv1 = nn.Conv2d(args.n_channels, outchannels, kernel_size=kernelsize, stride=stride, padding=padding)
        num_ftrs = self.model.fc.in_features
        self.feature_dim = num_ftrs
        self.model.fc = nn.Identity()  # Remove final classification layer

        self.fc = nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        embed = self.model(x)
        preds = self.fc(embed)
        return preds
    
    def train_model(self, train_loader):
        self.train()
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        milestones = [int(milestone) for milestone in self.args.milestones.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        # binary classification
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.75]).cuda() if torch.cuda.is_available() else torch.tensor([0.75]))
        best_acc , best_epoch = 0, -1

        for epoch in range(1, self.args.epochs + 1):
            self.train()
            epoch_loss = 0.0
            for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
                data, target = data.cuda(), target.cuda() if torch.cuda.is_available() else (data, target)
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output.squeeze(), target.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            logging.info(f"Epoch {epoch}/{self.args.epochs}, Loss: {avg_loss:.4f}")
            if epoch % self.args.eval_interval == 0:
                logging.info(f"Evaluation at epoch {epoch}:")
                acc = self.evaluate_model(train_loader)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    torch.save(self.state_dict(), f"{self.args.outdir}/best_model.pt")
                    logging.info(f"New best model saved with accuracy: {best_acc:.4f} at epoch {best_epoch}")
    
    def evaluate_model(self, val_loader):
        preds = []
        targets = []
        self.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda() if torch.cuda.is_available() else (data, target)
                output = self(data)
                preds += torch.sigmoid(output).cpu().numpy().flatten().tolist()
                targets += target.cpu().numpy().flatten().tolist()
        auc = roc_auc_score(targets, preds)
        preds = [1 if p >= 0.5 else 0 for p in preds]
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds)
        recall = recall_score(targets, preds)
        f1 = f1_score(targets, preds)
        sens = recall
        spec = recall_score(targets, preds, pos_label=0)

        logging.info(f"AUC: {auc:.4f}, Acc.: {accuracy:.4f}, Prec.: {precision:.4f}, Rec.: {recall:.4f}, F1: {f1:.4f}, Sens.: {sens:.4f}, Spec.: {spec:.4f}")
        return accuracy       

                
if __name__ == "__main__":
    class Args:
        pretrained=False
        vol_input_channels=36

    input_ = torch.randn(5, 36, 320, 320)
    args = Args()
    model = VolClsModel(args)
    out_ = model(input_)