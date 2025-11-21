import torch
import torch.nn as nn
from fft_conv import FFTConv2d
from torchvision.models import resnet18, resnet34, resnet50
import h5py

class KSpaceModel(nn.Module):
    def __init__(self, args):
        super(KSpaceModel, self).__init__()
        self.args = args
        
        if args.arch == "resnet18":
            self.model = resnet18(weights="IMAGENET1K_V1" if args.pretrained else None)
        elif args.arch == "resnet34":
            self.model = resnet34(weights="IMAGENET1K_V1" if args.pretrained else None)
        elif args.arch == "resnet50":
            self.model = resnet50(weights="IMAGENET1K_V1" if args.pretrained else None)
        
        num_ftrs = self.model.fc.in_features
        self.feature_dim = num_ftrs

        self.model.fc = nn.Identity()

        self.kspace_conv = FFTConv2d(1, 1, kernel_size=5, bias=True)

        self.slice_classifier = nn.Linear(num_ftrs, 1)

        if args.input_data_format == "slices+volumes":
            self.attention = nn.Sequential(
                self.slice_classifier,
                nn.Softmax(dim=0)
                )
            self.vol_classifier = nn.Linear(num_ftrs, 1)  # Binary classification

        self.layernorm = nn.LayerNorm(
            elementwise_affine=False, normalized_shape=(320, 320)
        )

    def slice_forward(self, x):

        kspace = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        kspace = torch.fft.fftn(kspace, dim=(-2, -1))
        out_complex = self.kspace_conv(kspace)        
        out_mag = out_complex.abs()
        
        out_mag = center_crop(out_mag, (320, 320))
        out_mag = self.layernorm(out_mag)
        
        if out_mag.shape[1] == 1:
            out = out_mag.repeat(1, 3, 1, 1)
        else:
            out = out_mag

        out = self.model(out)
        return self.slice_classifier(out), out  # return logits and features

    def vol_forward(self, x):
        return self.vol_classifier(x)

    def train_model(self, trainloader, valloader):
        self.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        vol_criterion = nn.BCEWithLogitsLoss()
        milestones = [int(milestone) for milestone in self.args.milestones.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_acc, best_epoch = 0, -1
        for epoch in range(self.args.num_epochs):
            self.train()
            epoch_loss = 0.0
            for i, (_, inputs, slice_labels, vol_labels_) in tqdm(enumerate(trainloader)):
                    # inputs has variable length inputs such as 5 x320 x 320, 32 x 320 x 320, 60 x 320 x 320, stack them into -1 x 1 x 320 x 320
                slices = torch.vstack([inp.unsqueeze(1) for inp in inputs])  # (total_slices, 1, height, width)

                # use torch.cat instead of torch.stack to avoid extra dimension
                slice_labels = slice_labels.view(-1, 1).float()  # (total_slices, 1)

                slices, slice_labels = slices.to(device), slice_labels.to(device)

                assert slices.shape[0] == slice_labels.shape[0], "Number of slices and labels must match"
                optimizer.zero_grad()
                slice_logits, slice_features = self.slice_forward(slices)
                slice_loss = criterion(slice_logits, slice_labels)
                total_loss = self.args.slice_loss_lam * slice_loss

                if self.args.input_data_format == "slices+volumes":
                    # Volume-level training
                    vol_preds = []
                    vol_labels = []
                    start_idx = 0
                    for inp, vol_label in zip(inputs, vol_labels_):
                        n_slices = inp.shape[0]
                        # reuse slice features from slice_forward
                        vol_slice_features = slice_features[start_idx:start_idx + n_slices]  # (n_slices, feature_dim)
                        start_idx += n_slices
                        # Compute attention weights
                        attn_weights = self.attention(vol_slice_features)  # (n_slices, 1)
                        # Weighted sum of slice features
                        topk_scores, topk_indices = torch.topk(attn_weights.squeeze(), k=5, largest=True)
                        topk_slice_features = vol_slice_features[topk_indices]
                        topk_attn_weights = topk_scores.unsqueeze(1)
                        vol_feature = torch.sum(topk_attn_weights * topk_slice_features, dim=0, keepdim=True)  # (1, feature_dim)
                        # normalize vol_feature
                        vol_feature = vol_feature / torch.sum(topk_attn_weights)
                        # vol_feature = torch.sum(attn_weights * vol_slice_features, dim=0, keepdim=True)  # (1, feature_dim)
                        vol_logit = self.vol_forward(vol_feature)  # (1, 1)
                        vol_preds.append(vol_logit)

                        vol_label = vol_label.to(device)
                        vol_labels.append(vol_label)
                    vol_preds = torch.vstack(vol_preds).cuda()  # (batch_size, 1)
                    vol_labels = torch.tensor(vol_labels).unsqueeze(1).float().cuda()  # (batch_size, 1)
                    vol_loss = vol_criterion(vol_preds, vol_labels)
                    total_loss += vol_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                print(f"step {i+1}/{len(trainloader)}, loss: {epoch_loss/(i+1):.4f}")
            acc, prec, rec, f1, roc, sens, spec = self.eval_model(valloader)  # evaluate on validation set each iteration
            scheduler.step()
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                torch.save(self.state_dict(), f"{self.args.outdir}/best_model.pth")
                logging.info(f"New best model saved at epoch {best_epoch} with accuracy {best_acc:.4f}")
            
            avg_loss = epoch_loss / len(trainloader)
            logging.info(f"epoch {epoch+1}/{self.args.epochs}, loss: {avg_loss:.4f}")
        logging.info("Finished Slice Model Training")
        torch.save(self.state_dict(), f"{self.args.outdir}/final_model.pth")            

def center_crop(data, shape) -> torch.Tensor:
    """
    Center crop of data. 
    Args:
        data: torch.Tensor
        shape: Tuple[int, int]
    Returns:
        torch.Tensor        
    """    
    if data.shape[-2:] == shape:
        return data

    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.arch = "resnet18"
            self.pretrained = True
            self.input_data_format = "slices"
    args = Args()
    model = KSpaceModel(args)

    with h5py.File("../../../file1000891.h5", "r") as f:
        kspace = f["kspace"][:]
    kspace = torch.from_numpy(kspace).unsqueeze(1).to(torch.complex64)
    print(kspace.shape)

    output = model.slice_forward(kspace)
    print(output)