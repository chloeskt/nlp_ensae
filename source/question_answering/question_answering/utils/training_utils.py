from typing import Tuple, Any, List, Union

import datasets
import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    CanineForQuestionAnswering,
    PreTrainedTokenizer,
)

from .utils import compute_metrics, postprocess_qa_predictions


def train_canine(
    model: CanineForQuestionAnswering,
    num_epochs: int,
    optimizer: torch.optim.AdamW,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dataset: datasets.arrow_dataset.Dataset,
    features_val_dataset: datasets.arrow_dataset.Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    metric: Any,
    learning_rate: float,
    max_answer_length: int,
    n_best_size: int,
    output_dir: str,
    best_f1: float = np.Inf,
    lr_scheduler=None,
    drive: bool = True,
    squad_v2: bool = True,
    clipping: bool = True,
) -> Tuple[List[float], List[float]]:
    # setup GPU/CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # move model over to detected device
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        nb_iterations = 0
        running_loss = 0.0
        model.train()
        loop = tqdm(train_loader)

        for batch in loop:
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # pull all the tensor batches required for training
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()

            if clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            # update parameters
            optimizer.step()
            # print relevant info to progress bar
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())
            running_loss += loss.item()

            if lr_scheduler:
                lr_scheduler.step()

                nb_iterations += 1

        val_acc, val_loss, f1, exact_match = evaluate(
            model,
            val_dataset,
            features_val_dataset,
            val_loader,
            tokenizer,
            device,
            batch_size,
            metric,
            max_answer_length,
            n_best_size,
            squad_v2,
        )  # Compute validation loss
        print()
        print(
            "Epoch {} complete! Training Loss: {}, Validation Loss : {}, Validation Accuracy: {}".format(
                epoch, running_loss / nb_iterations, val_loss, val_acc
            )
        )
        print("F1-score: {}, Exact match: {}".format(f1, exact_match))

        train_losses.append(running_loss / nb_iterations)
        val_losses.append(val_loss)

        if f1 > best_f1:
            print("Best F1_score improved from {} to {}".format(best_f1, f1))
            best_f1 = f1
            best_epoch = epoch

            # Saving the model
            path_to_model = f"{output_dir}/CANINE_lr_{learning_rate}_val_loss_{val_loss}_f1_{best_f1}_acc_{val_acc}_ep_{best_epoch}.pt"
            if drive:
                path_to_model = f"/content/drive/MyDrive/{output_dir}/CANINE_lr_{learning_rate}_val_loss_{val_loss}_f1_{best_f1}_acc_{val_acc}_ep_{best_epoch}.pt"
            torch.save(model.state_dict(), path_to_model)
            print("The model has been saved in {}".format(path_to_model))
            print("\n")

    del loss
    torch.cuda.empty_cache()
    return train_losses, val_losses


def evaluate(
    model: CanineForQuestionAnswering,
    val_dataset: datasets.arrow_dataset.Dataset,
    features_val_dataset: datasets.arrow_dataset.Dataset,
    val_loader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    device: str,
    batch_size: int,
    metric: Any,
    max_answer_length: int,
    n_best_size: int,
    squad_v2: bool = True,
) -> Tuple[Union[float, Any], float, ndarray, ndarray]:
    # switch model out of training mode
    model.eval()
    # initialize list to store accuracies, f1-score, exact matches
    acc = []
    f1_scores = []
    exact_matches = []

    # loop through batches
    running_loss = 0.0
    loop = tqdm(val_loader)

    nb_iterations = 0
    data_batch_min_index = 0
    last_data_id = 0

    for i, batch in enumerate(loop):
        features_dataset = features_val_dataset[i * batch_size : batch_size * (i + 1)]
        examples_id_in_features_dataset = features_dataset["example_id"]
        start_index = data_batch_min_index
        indices_to_consider = []
        while val_dataset[data_batch_min_index][
            "id"
        ] in examples_id_in_features_dataset and data_batch_min_index < len(
            val_dataset
        ):
            indices_to_consider.append(data_batch_min_index)
            data_batch_min_index += 1

        if examples_id_in_features_dataset[0] == last_data_id:
            start_index -= 1

        data = val_dataset[start_index:data_batch_min_index]

        if set(examples_id_in_features_dataset) == set(data["id"]):
            pass
        else:
            print("KO", examples_id_in_features_dataset, data["id"])

        last_data_id = data["id"][-1]

        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            # we will use true positions for accuracy calc
            start_true = batch["start_positions"].to(device)
            end_true = batch["end_positions"].to(device)
            # make predictions
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_true,
                end_positions=end_true,
            )
            # extract loss
            loss = outputs.loss
            running_loss += loss.item()
            # pull prediction tensors out and argmax to get predicted tokens
            start_pred = outputs.start_logits.argmax(dim=-1)
            end_pred = outputs.end_logits.argmax(dim=-1)
            # calculate accuracy for both and append to accuracy list
            acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
            acc.append(((end_pred == end_true).sum() / len(end_pred)).item())
            loop.set_postfix(loss=loss.item())

            # calculate exact match and f1-score metrics for squad dataset
            predictions, data_val = postprocess_qa_predictions(
                data,
                features_dataset,
                outputs,
                tokenizer,
                n_best_size=n_best_size,
                max_answer_length=max_answer_length,
                squad_v2=squad_v2,
            )

            metrics = compute_metrics(
                metric,
                data_val,
                predictions,
                squad_v2,
            )

            f1_scores.append(metrics["f1"])
            if squad_v2:
                exact_matches.append(metrics["exact"])
            else:
                exact_matches.append(metrics["exact_match"])

            nb_iterations += 1

    # calculate average accuracy in total
    acc = sum(acc) / len(acc)
    return (
        acc,
        (running_loss / nb_iterations),
        np.mean(f1_scores),
        np.mean(exact_matches),
    )
