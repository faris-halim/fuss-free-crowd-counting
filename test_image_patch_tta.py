import os
import argparse
import torch
import numpy as np

from Networks import FFNet
from datasets import crowd
from torch.utils.data import DataLoader


def test(args):

    # -----------------------------
    # 1. Select CPU as our device
    # -----------------------------
    device = torch.device("cpu")

    print("Using device:", device)

    # -----------------------------
    # 2. Load ShanghaiTech Part B
    # -----------------------------
    data_path = args.data_path

    dataset = crowd.Crowd_sh(
        os.path.join(data_path, "test_data"),
        args.crop_size,
        8,
        method="val"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Test images:", len(dataset))

    # -----------------------------
    # 3. Create FFNet model
    # -----------------------------
    model = FFNet.FFNet()

    model = model.to(device)

    # -----------------------------
    # 4. Load pretrained model
    # -----------------------------
    print("Loading pretrained model...")

    model.load_state_dict(
        torch.load(
            args.model_path,
            map_location=device
        )
    )

    model.eval()

    print("Model loaded successfully.")

    # -----------------------------
    # 5. Test the model
    # -----------------------------
    errors = []

    with torch.no_grad():
        for i, (img, actual_count, filename) in enumerate(dataloader):

            img = img.to(device)

            # --------------------------------
            # Test-Time Augmentation (TTA)
            # --------------------------------

            # 1. Prediction from original image
            output_original = model(img)[0]

            # 2. Flip the image horizontally
            flipped_img = torch.flip(img, dims=[3])

            # 3. Prediction from flipped image
            output_flipped = model(flipped_img)[0]

            # 4. Flip the predicted density map back
            output_flipped = torch.flip(output_flipped, dims=[3])

            # 5. Average both density maps
            output = (output_original + output_flipped) / 2

            # 6. Convert final density map into crowd count
            pred_count = output.sum().item()

            # Actual crowd count
            actual_count = actual_count.item()

            # Calculate absolute error
            error = abs(pred_count - actual_count)

            errors.append(error)

            print(
                f"Image {i + 1} ({filename[0]}): "
                f"Actual = {actual_count:.2f}, "
                f"Predicted = {pred_count:.2f}, "
                f"Error = {error:.2f}"
            )

    # -----------------------------
    # 6. Calculate MAE
    # -----------------------------
    mae = np.mean(errors)

    # -----------------------------
    # 7. Calculate MSE
    # -----------------------------
    mse = np.mean(np.square(errors))

    print("\n==============================")
    print("TESTING COMPLETE")
    print("==============================")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
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
    


