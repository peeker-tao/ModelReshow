import torch
import numpy as np
import pandas as pd

# def evaluate(model, test_loader, criterion, device):
#     model.to(device)
#     model.eval()
#     test_loss = 0.0
#     test_count = 0.0
#     with torch.no_grad():
#         for images, keypoints in test_loader:
#             images, keypoints = images.to(device), keypoints.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, keypoints)
#             test_loss += loss.item() * images.size(0)
#             test_count += images.size(0)

#     test_loss /= len(test_loader.dataset)
#     test_accuracy = test_count / len(test_loader.dataset)
#     return test_loss, test_accuracy


DEFAULT_FEATURE_NAMES = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "left_eye_inner_corner_x",
    "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",
    "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",
    "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",
    "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",
    "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",
    "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",
    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",
    "right_eyebrow_outer_end_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_left_corner_x",
    "mouth_left_corner_y",
    "mouth_right_corner_x",
    "mouth_right_corner_y",
    "mouth_center_top_lip_x",
    "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",
    "mouth_center_bottom_lip_y",
]


def predict(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            outputs = model(images)
            predictions.append(outputs.cpu().numpy())

    if not predictions:
        return np.empty((0, 0), dtype=np.float32)
    predictions = np.concatenate(predictions, axis=0)
    return np.clip(predictions, 0.0, 96.0)


def make_submission(predictions, lookup_csv, output_csv, feature_names = DEFAULT_FEATURE_NAMES, ):
    lookup = pd.read_csv(lookup_csv)

    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")
    if predictions.shape[1] != len(feature_names):
        raise ValueError(
            f"prediction dim {predictions.shape[1]} does not match feature count {len(feature_names)}"
        )

    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    locations = []
    for _, row in lookup.iterrows():
        image_id = int(row["ImageId"]) - 1
        feature_name = row["FeatureName"]
        if feature_name not in feature_to_idx:
            raise KeyError(f"Unknown feature name: {feature_name}")
        if image_id < 0 or image_id >= len(predictions):
            raise IndexError(
                f"ImageId {image_id + 1} out of range for predictions with length {len(predictions)}"
            )
        locations.append(predictions[image_id, feature_to_idx[feature_name]])

    submission = pd.DataFrame(
        {
            "RowId": lookup["RowId"],
            "Location": locations,
        }
    )
    submission.to_csv(output_csv, index=False)
    return submission


def make_prediction_table(predictions, lookup_csv, output_csv, feature_names=DEFAULT_FEATURE_NAMES):
    lookup = pd.read_csv(lookup_csv)

    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")
    if predictions.shape[1] != len(feature_names):
        raise ValueError(
            f"prediction dim {predictions.shape[1]} does not match feature count {len(feature_names)}"
        )

    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    rows = []
    for _, row in lookup.iterrows():
        image_id = int(row["ImageId"]) - 1
        feature_name = row["FeatureName"]
        if feature_name not in feature_to_idx:
            raise KeyError(f"Unknown feature name: {feature_name}")
        if image_id < 0 or image_id >= len(predictions):
            raise IndexError(
                f"ImageId {image_id + 1} out of range for predictions with length {len(predictions)}"
            )
        rows.append(
            {
                "RowId": int(row["RowId"]),
                "ImageId": int(row["ImageId"]),
                "FeatureName": feature_name,
                "Location": float(predictions[image_id, feature_to_idx[feature_name]]),
            }
        )

    prediction_table = pd.DataFrame(rows)
    prediction_table.to_csv(output_csv, index=False)
    return prediction_table


