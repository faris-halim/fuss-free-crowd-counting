import os
import argparse
import torch
import numpy as np

from Networks import FFNet
from datasets import crowd
from torch.utils.data import DataLoader


def get_predictions(model, dataset_path, crop_size, device):

    dataset = crowd.Crowd_sh(
        dataset_path,
        crop_size,
        8,
        method="val"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    predictions = []
    actuals = []

    with torch.no_grad():

        for img, actual_count, filename in dataloader:

            img = img.to(device)

            # Model prediction
            output = model(img)

            # Density map ka total = predicted crowd count
            pred_count = output[0].sum().item()

            actual_count = actual_count.item()

            predictions.append(pred_count)
            actuals.append(actual_count)

    return np.array(predictions), np.array(actuals)


def test(args):

    # --------------------------------
    # 1. Use CPU
    # --------------------------------

    device = torch.device("cpu")

    print("Using device:", device)


    # --------------------------------
    # 2. Create model
    # --------------------------------

    model = FFNet.FFNet()

    model = model.to(device)


    # --------------------------------
    # 3. Load pretrained model
    # --------------------------------

    print("Loading pretrained model...")

    model.load_state_dict(
        torch.load(
            args.model_path,
            map_location=device
        )
    )

    model.eval()

    print("Model loaded successfully.")


    # ==========================================
    # PART A — CALCULATE CALIBRATION FACTOR
    # ==========================================

    print("\nCalculating calibration factor from TRAIN data...")


    train_path = os.path.join(
        args.data_path,
        "train_data"
    )


    train_predictions, train_actuals = get_predictions(
        model,
        train_path,
        args.crop_size,
        device
    )


    # Total actual people
    total_actual = train_actuals.sum()

    # Total predicted people
    total_predicted = train_predictions.sum()


    # Calibration factor
    calibration_factor = total_actual / total_predicted


    print("\nCalibration factor:", calibration_factor)


    # ==========================================
    # PART B — TEST ORIGINAL MODEL
    # ==========================================

    print("\nTesting ORIGINAL model...")


    test_path = os.path.join(
        args.data_path,
        "test_data"
    )


    test_predictions, test_actuals = get_predictions(
        model,
        test_path,
        args.crop_size,
        device
    )


    original_errors = np.abs(
        test_predictions - test_actuals
    )


    original_mae = np.mean(original_errors)

    original_mse = np.mean(
        original_errors ** 2
    )


    # ==========================================
    # PART C — APPLY CALIBRATION
    # ==========================================

    print("\nApplying calibration...")


    calibrated_predictions = (
        test_predictions * calibration_factor
    )


    calibrated_errors = np.abs(
        calibrated_predictions - test_actuals
    )


    calibrated_mae = np.mean(
        calibrated_errors
    )


    calibrated_mse = np.mean(
        calibrated_errors ** 2
    )


    # ==========================================
    # PART D — SHOW SOME RESULTS
    # ==========================================

    print("\n==============================")
    print("CALIBRATION RESULTS")
    print("==============================")


    print(
        f"Calibration factor: "
        f"{calibration_factor:.4f}"
    )


    print(
        f"\nOriginal MAE: "
        f"{original_mae:.4f}"
    )


    print(
        f"Calibrated MAE: "
        f"{calibrated_mae:.4f}"
    )


    print(
        f"\nOriginal MSE: "
        f"{original_mse:.4f}"
    )


    print(
        f"Calibrated MSE: "
        f"{calibrated_mse:.4f}"
    )


    print("\nSample predictions:")


    for i in range(min(10, len(test_actuals))):

        print(
            f"Image {i + 1}: "
            f"Actual = {test_actuals[i]:.2f}, "
            f"Original = {test_predictions[i]:.2f}, "
            f"Calibrated = {calibrated_predictions[i]:.2f}"
        )


    print("\n==============================")
    print("CALIBRATION COMPLETE")
    print("==============================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--data-path",
        default="./part_B_final"
    )


    parser.add_argument(
        "--model_path",
        default="./SHB_model.pth"
    )


    parser.add_argument(
        "--crop-size",
        type=int,
        default=512
    )


    args = parser.parse_args()

    test(args)