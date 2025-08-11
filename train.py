from dataclasses import dataclass
import os
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.generate.classifier_model import WideResNet_noisy
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
from src.utils.data_loader import get_train_valid_loader, get_test_loader
from src.utils.load_unlabeld import make_unlabeled_loader_from_manifest, ManifestDataset, CombineManifestDataset
import json

from src.generate.hyperparams import train_params

upper_limit, lower_limit = 1,0
epsilon = (8 / 255.)
pgd_alpha = (2 / 255.)
attack_iters_train = 10


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(X).to(y.device)

    for _ in range(restarts):
        delta = torch.zeros_like(X).to(y.device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        
        for _ in range(attack_iters):
            output = model(X + delta)
            index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_clean_accuracy(model, test_dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input, target in test_dataloader:
            inputs = input.to(device)
            labels = target.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    clean_acc = 100.0 * correct / total
    print("Clean Test Accuracy:",clean_acc)
    return clean_acc


def evaluate_adversarial_accuracy(model, test_dataloader, attack_pgd, device):
    model.eval()
    correct = 0
    total = 0

    for batch in test_dataloader:
        inputs = batch['input'].to(device)
        labels = batch['target'].to(device)

        # Generate adversarial perturbations
        delta = attack_pgd(
            model=model,
            X=inputs,
            y=labels,
            epsilon=8/255,
            alpha=0.1,
            attack_iters=20,
            restarts=1,
            norm="l_inf"
        )

        # Evaluate model on adversarial examples
        with torch.no_grad():
            adv_inputs = torch.clamp(inputs + delta, 0, 1)
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    adv_acc = 100.0 * correct / total
    print("Adversarial Test Accuracy:", adv_acc)
    return adv_acc


def ssat_train(model,
            train_dataloader_l,
            test_dataloader_l,
            steps_per_epoch,
            batch_size, 
            optimizer, 
            criterion, 
            device, 
            normalize, 
            epochs, 
            alpha, 
            attack_iters, 
            epsilon,
            B_u,
            ds,
            save_every=25,
            save_path="./robust_classifier/",
            eval_params=None):
            
    from tqdm import tqdm  # Import tqdm for progress bars
    model.train()
    model = model.to(device)

    # Define the number of data points per epoch
    points_per_epoch = 50000
    batches_per_epoch = points_per_epoch // batch_size

    clean_acc_list = []
    adv_acc_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # load new batches for each epoch from unlabeled manifest
        train_u_loader = make_unlabeled_loader_from_manifest(int(steps_per_epoch), int(B_u), ds)

        for i, ((X_labeled, y_labeled), (X_unlabeled, y_unlabeled)) in enumerate(tqdm(zip(train_u_loader, train_dataloader_l))):
            if i >= batches_per_epoch:
                break

            # Get labeled data
            X_labeled, y_labeled = X_labeled.to(device), y_labeled.to(device)

            # Get unlabeled data (pseudo-labeled)
            X_unlabeled, y_unlabeled = X_unlabeled.to(device), y_unlabeled.to(device)

            # Combine both
            X = torch.cat([X_labeled, X_unlabeled], dim=0)
            y = torch.cat([y_labeled, y_unlabeled], dim=0)

            # Generate adversarial perturbation
            delta = attack_pgd(
                model, X, y,
                epsilon=epsilon,
                alpha=alpha,
                attack_iters=attack_iters,
                restarts=1,
                norm="l_inf"
            )
            adv_X = torch.clamp(X + delta, 0, 1)

            # Forward + Backward + Optimize
            outputs = model(adv_X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item() * y.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/total:.4f}, Accuracy: {100.0 * correct / total:.2f}%")

        # Save model and evaluate
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            
            # Evaluate on clean and adversarial test sets
            clean_acc = evaluate_clean_accuracy(model, test_dataloader_l, device)
            adv_acc = evaluate_adversarial_accuracy(model, test_dataloader_l, attack_pgd, device)
            clean_acc_list.append(clean_acc)
            adv_acc_list.append(adv_acc)
            
            # Save the model
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))
            print(f"Model saved at {os.path.join(save_path, f'model_epoch_{epoch+1}.pth')}")

            # save accuracies as json
            with open(os.path.join(save_path, 'accuracies.json'), 'w') as f:
                json.dump({
                    'clean_accuracies': clean_acc_list,
                    'adversarial_accuracies': adv_acc_list,
                    "epoch": epoch + 1,
                }, f)

            model.train()


        # Print epoch statistics
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


if __name__ == "__main__":
    
    # define device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # initialize model
    robust_classifier = WideResNet_noisy(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
    robust_classifier = torch.nn.DataParallel(robust_classifier)
    robust_classifier.to(device)

    # load clean batches with mixed generation to real ratio
    batch_size = 256; B_l = int(batch_size * 0.5); B_u = int(batch_size * 0.5)
    steps_per_epoch = 50000 // batch_size  # Assuming batch size of 128

    train_dataloader_l, valid_dataloader_l = get_train_valid_loader("./data/clean",batch_size=B_l, valid_size=0.1, shuffle=True, num_workers=8, pin_memory=True, augment=False, random_seed=0)
    test_dataloader_l = get_test_loader("./data/clean", batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # load unlabeled batches, build manifest datasets
    manifest_root_1 = "./conditional_generated_data/20250821-143115"
    manifest_root_2 = "./conditional_generated_data/20250821-142835"
    print("Building manifest data from", manifest_root_1, "and", manifest_root_2)
    alpha = 0.5; beta = 0.5
    mmin = 0.0; mmax = 1; tau_keep = 0
    ds = CombineManifestDataset(
        manifest_root_1, manifest_root_2,
        mmin=mmin, mmax=mmax, tau_keep=tau_keep,
        use_pred_if_no_target=True,
        alpha=alpha, beta=beta
    )
    print(f"Loaded {len(ds)} items from manifest datasets with alpha={alpha} and beta={beta}")

    print("Training...")
    epochs = 250  # Adjust as needed
    robust_classifier.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ssat_train(
        model=robust_classifier,
        train_dataloader_l=train_dataloader_l,
        test_dataloader_l=test_dataloader_l,
        steps_per_epoch=train_params.steps_per_epoch,
        batch_size=train_params.batch_size,
        optimizer=optim.Adam(robust_classifier.parameters(), lr=train_params.lr),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        normalize=True,
        epochs=train_params.epochs,
        alpha=train_params.alpha_step,
        attack_iters=train_params.attack_iters,
        epsilon=train_params.epsilon,
        B_u=int(train_params.batch_size * 0.5),
        ds=ds
    )
    print("Training complete.")