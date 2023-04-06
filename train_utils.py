import json
import torch
import time
import copy


def train(num_epoches, model, loss_fn, optimizer, dataset_name, train_dataloader, validation_dataloader,
          scaler, logger, best_model_path, loss_save_path, early_stop=10):
    """
        Train the model for a given number of epochs.

        If the validation loss does not improve for a given number of early_stop epochs,
        the training process will be stopped.
    """

    logger.info(
        f"================== Start training for {num_epoches} epochs for {dataset_name} dataset. ==================")
    best_model = None
    best_loss = float('inf')
    not_improved_count = 0
    train_loss_list = []
    val_loss_list = []

    start_time = time.time()
    for epoch in range(1, num_epoches + 1):
        train_epoch_loss = train_one_epoch(
            epoch, model, train_dataloader, optimizer, scaler,  loss_fn, logger)
        val_epoch_loss = validate(
            epoch, model, validation_dataloader, scaler, loss_fn, logger)

        train_loss_list.append(train_epoch_loss)
        val_loss_list.append(val_epoch_loss)
        if train_epoch_loss > 1e6:
            logger.error(
                "Something wrong with training process. Loss exploded.")
            break
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            not_improved_count = 0
            best_state = True  # save the this model as the best model
        else:
            not_improved_count += 1
            best_state = False  # do not save current model as the best model
        if not_improved_count == early_stop:
            logger.info(f"Validation performance didn\'t improve for {not_improved_count} epochs."
                        "Training stops.")
            break
        if best_state == True:
            logger.info(
                ">> Copying current model as the best model due to the lower validation loss.")
            best_model = copy.deepcopy(model.state_dict())

    training_time = time.time() - start_time
    logger.info(
        f"Total training time: {round(training_time / 60,4)}min, best loss: {round(best_loss,5)}")
    
    torch.save(best_model, best_model_path)
    logger.info(f">> The last best model is now saved to: {best_model_path}")

    # saving the loss curves
    with open(loss_save_path, "w") as file:
        json.dump({"train": train_loss_list, "validation": val_loss_list}, file)
    logger.info(f">> The loss curves are now saved to: {loss_save_path}")


def validate(epoch, model, val_dataloader, scaler, loss_fn, logger):

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for _, (data, target) in enumerate(val_dataloader):
            data = data[..., :1]
            label = target[..., :1]
            output = model(data)
            label = scaler.inverse_transform(label)
            loss = loss_fn(output.cuda(), label)
            if not torch.isnan(loss):
                total_val_loss += loss.item()

    average_val_epoch_loss = total_val_loss / len(val_dataloader)
    logger.info(f">> Validation Epoch {epoch} Finished. Averaged Validation Epoch Loss: {round(average_val_epoch_loss, 6)}")

    return average_val_epoch_loss





def train_one_epoch(epoch, model, train_dataloader, optimizer, scaler,  loss_fn, logger):
    total_batches = len(train_dataloader)

    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data[..., :1]
        label = target[..., :1]  # (..., 1)
        optimizer.zero_grad()

        # data and target shape: B, T, N, F; output shape: B, T, N, F
        output = model(data).cuda()
        label = scaler.inverse_transform(label)
        loss = loss_fn(output, label)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # log information every 50 batches
        if batch_idx % 50 == 0:
            logger.info(
                f"Train Epoch {epoch}: {batch_idx}/{total_batches} Loss: {round(loss.item(), 6)} ")

    average_epoch_loss = total_loss/total_batches
    logger.info(
        f">> Train Epoch {epoch} Finished. Averaged Train Epoch Loss: {round(average_epoch_loss, 6)}")

    return average_epoch_loss
