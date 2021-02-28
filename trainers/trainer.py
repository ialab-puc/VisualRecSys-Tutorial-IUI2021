import copy
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.utils import get_cpu_copy, save_checkpoint


class Trainer:

    def __init__(
            self, model, device, criterion, optimizer, scheduler,
            checkpoint_dir=None, writer_dir=None, version=None,
        ):
        # Important objects for training
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Device
        if device is None:
            device = torch.device("cpu")
            print(f">> Device is none... default: {device}")
        self.device = device
        self.model = self.model.to(self.device)

        # Version
        if version is None:
            version = (
                f"{model.__class__.__name__}_"
                f"{time.strftime('%Y-%m-%d-%H-%M-%S')}"
            )
        self.version = version
        print(f">> Model version: {self.version}")

        # Checkpoints
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join("checkpoints")
        assert os.path.isdir(checkpoint_dir)
        self.checkpoint_dst = os.path.join(checkpoint_dir, f"{self.version}.tar")
        print(f">> Checkpoints stored at... {self.checkpoint_dst}")

    def run(
            self, max_epochs, max_learning_rates, dataloaders,
            non_blocking=True, train_valid_loops=1, save_last_model=False,
        ):
        # Prepare model
        self.model = self.model.to(self.device)

        # Save first checkpoint
        save_checkpoint(
            # Base values
            self.checkpoint_dst, model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            # Epoch values
            epoch=None, accuracy=None, loss=None,
        )

        # Starting values
        best_validation_acc = 0.0
        best_validation_loss = float("inf")
        used_lrs = [self.optimizer.param_groups[0]["lr"]]

        # Measure elapsed time
        start = time.time()

        # Progress bars
        assert all(key in dataloaders for key in ["train", "validation"])
        pbar_epochs = tqdm(
            total=max_epochs,
            desc="Epoch", unit="epoch",
            postfix={
                "current_lr": used_lrs[0],
                "used_lrs": len(used_lrs),
            },
        )
        pbar_train = tqdm(
            total=train_valid_loops * len(dataloaders["train"]),
            desc="Train",
            postfix={
                "last_acc": None,
                "last_lostt": None,
            },
        )
        pbar_valid = tqdm(
            total=len(dataloaders["validation"]),
            desc="Valid",
            postfix={
                "best_acc": None,
                "best_loss": None,
                "best_epoch": None,
                "bad_epochs": f"{self.scheduler.num_bad_epochs}",
            },
        )

        # Training loop
        for epoch in range(1, max_epochs + 1):
            # Each epoch has a training and a validation phase
            for phase in ["train", "validation"]:
                # Update model mode and progress bar
                if phase == "train":
                    self.model.train()
                    pbar_train.reset()
                    pbar_valid.reset()
                elif phase == "validation":
                    self.model.eval()

                # Value accumulators
                running_acc = torch.tensor(0, dtype=int, device=self.device)
                running_loss = torch.tensor(0.0, dtype=torch.double, device=self.device)

                # Iterate over data
                dataset = dataloaders[phase].dataset
                loop_times = train_valid_loops if phase == "train" else 1
                for _ in range(loop_times):
                    for i_batch, data in enumerate(dataloaders[phase]):
                        profile = data[0].to(self.device, non_blocking=non_blocking).squeeze(dim=0)
                        pi = data[1].to(self.device, non_blocking=non_blocking).squeeze(dim=0)
                        ni = data[2].to(self.device, non_blocking=non_blocking).squeeze(dim=0)
                        target = torch.ones(pi.size(0), 1, 1, device=self.device)

                        # Restart params gradients
                        self.optimizer.zero_grad()

                        # Forward pass
                        with torch.set_grad_enabled(phase == "train"):
                            output = self.model(profile, pi, ni)
                            loss = self.criterion(output, target)
                            # Backward pass
                            if phase == "train":
                                loss.backward()
                                self.optimizer.step()

                        # Statistics
                        running_acc.add_((output > 0).sum())
                        running_loss.add_(loss.detach() * output.size(0))

                        # Update progress bar
                        if phase == "train":
                            pbar_train.update()
                        else:
                            pbar_valid.update()

                        # Synchronize GPU (debugging)
                        # torch.cuda.synchronize()

                # Aggregate statistics
                dataset_size = loop_times * len(dataset)
                epoch_acc = running_acc.item() / dataset_size
                epoch_loss = running_loss.item() / dataset_size
                # tqdm.write(f">> Epoch {epoch} ({phase.title()}) | ACC {100 * epoch_acc:.3f} - Loss {epoch_loss:.6f}")

                if phase == "train":
                    # Update progress bar
                    pbar_train.set_postfix({
                        "last_acc": f"{100 * epoch_acc:.3f}",
                        "last_loss": f"{epoch_loss:.6f}",
                    })
                elif phase == "validation":
                    new_optimal = False
                    if self.scheduler.mode == "max":
                        # Is this a new best accuracy?
                        new_optimal = epoch_acc > best_validation_acc
                    else:
                        # Is this a new best loss?
                        new_optimal = epoch_loss < best_validation_loss
                    if new_optimal:
                        # Save best model
                        best_validation_acc = epoch_acc
                        best_validation_loss = epoch_loss
                        save_checkpoint(
                            # Base values
                            self.checkpoint_dst, model=get_cpu_copy(self.model),
                            criterion=self.criterion,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            # Epoch values
                            epoch=self.scheduler.last_epoch,
                            accuracy=best_validation_acc,
                            loss=best_validation_loss,
                        )
                        # tqdm.write(f">> New best model (Epoch: {epoch}) | ACC {100 * epoch_acc:.3f} ({epoch_acc})")
                    # Scheduler step
                    if self.scheduler.mode == "max":
                        self.scheduler.step(epoch_acc)
                    else:
                        self.scheduler.step(epoch_loss)
                    next_lr = self.optimizer.param_groups[0]["lr"]
                    if next_lr not in used_lrs:
                        # tqdm.write(f">> Next lr: {next_lr} (Already used {used_lrs})")
                        used_lrs.append(next_lr)
                        pbar_epochs.set_postfix({
                            "used_lrs": len(used_lrs),
                            "current_lr": next_lr,
                        })
                    # Update progress bar
                    pbar_valid.set_postfix({
                        "best_acc": f"{100 * best_validation_acc:.3f}",
                        "best_loss": f"{best_validation_loss:.6f}",
                        "best_epoch": f"{epoch}",
                        "bad_epochs": f"{self.scheduler.num_bad_epochs}",
                    })

            # Update epochs pbar at the end
            pbar_epochs.update()
            # tqdm.write("\n")

            # Check if used all available learning rates
            if len(used_lrs) > max_learning_rates:
                print(f">> Reached max different lrs ({max_learning_rates}: {used_lrs})")
                break

        # Complete progress bars
        pbar_epochs.close()
        pbar_train.close()
        pbar_valid.close()

        # Report status
        elapsed = time.time() - start
        print(f">> Training completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
        print(f">> Best validation accuracy: ~{100 * best_validation_acc:.3f}%")
        print(f">> Best validation loss: ~{best_validation_loss:.6f}")

        if save_last_model:
            # Copy last model weights
            print(">> Copy last model")
            last_model_weights = copy.deepcopy(get_cpu_copy(self.model))
        else:
            epoch_acc = None
            epoch_loss = None
            last_model_weights = None

        # Load best model weights
        print(">> Load best model")
        best_checkpoint = torch.load(self.checkpoint_dst, map_location=torch.device("cpu"))
        self.model.load_state_dict(best_checkpoint["model"])

        # Move model back to device
        self.model.to(self.device)

        # Save last state
        print(">> Save last state")
        save_checkpoint(
            # Base values
            self.checkpoint_dst, model=get_cpu_copy(self.model),
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            # Epoch values
            epoch=self.scheduler.last_epoch,
            accuracy=best_validation_acc,
            loss=best_validation_loss,
            # Last values
            last_model=last_model_weights,
            last_accuracy=epoch_acc,
            last_loss=epoch_loss,
        )

        return self.model, best_checkpoint["accuracy"], best_checkpoint["loss"], best_checkpoint["epoch"]
