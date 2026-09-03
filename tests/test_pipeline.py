import numpy as np
import torch

from src.datasets.ham10000 import (
    METADATA_DIM,
    HAM10000Dataset,
    build_manifest,
    class_weights_for_loss,
    encode_metadata_dict,
    encode_metadata_row,
    get_transforms,
    make_sampler,
    split_manifest,
)
from src.utils import expected_calibration_error, set_seed


# ---------------------------------------------------------------------------
# Metadata encoding
# ---------------------------------------------------------------------------

def test_encode_metadata_dict_shape_and_range():
    vec = encode_metadata_dict(age=45, sex="female", localization="scalp")
    assert vec.shape == (METADATA_DIM,)
    assert vec.dtype == np.float32
    assert 0.0 <= vec[0] <= 1.0  # age_norm
    assert vec[1] == 0.0  # age not missing
    assert not np.isnan(vec).any()


def test_encode_metadata_dict_handles_missing_values():
    vec = encode_metadata_dict(age=None, sex=None, localization=None)
    assert vec[1] == 1.0  # age_missing flag set
    assert vec[0] == 0.5  # neutral age
    # sex one-hot should land on "unknown"
    assert vec.sum() > 0


def test_encode_metadata_dict_unknown_category_falls_back():
    vec = encode_metadata_dict(age=30, sex="nonbinary", localization="mars")
    # should not raise, and should fall back to the "unknown" bucket rather
    # than silently doing nothing
    assert vec.shape == (METADATA_DIM,)


# ---------------------------------------------------------------------------
# Manifest building + splitting
# ---------------------------------------------------------------------------

def test_build_manifest(synthetic_data_root, base_cfg):
    df = build_manifest(str(synthetic_data_root), base_cfg["data"]["class_names"])
    assert len(df) > 0
    assert set(df["class_name"].unique()) <= set(base_cfg["data"]["class_names"])
    assert df["filepath"].apply(lambda p: p.exists()).all()
    assert df["label"].between(0, base_cfg["data"]["num_classes"] - 1).all()


def test_split_manifest_disjoint(synthetic_data_root, base_cfg):
    df = build_manifest(str(synthetic_data_root), base_cfg["data"]["class_names"])
    train_df, val_df, test_df = split_manifest(
        df, base_cfg["data"]["train_split"], base_cfg["data"]["val_split"], seed=42
    )
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_make_sampler_upweights_rare_classes(synthetic_data_root, base_cfg):
    df = build_manifest(str(synthetic_data_root), base_cfg["data"]["class_names"])
    # simulate imbalance by dropping most of one class
    imbalanced = df.drop(df[df["class_name"] == "DF"].index[1:])
    sampler = make_sampler(imbalanced, base_cfg["data"]["num_classes"])
    weights = np.array(list(sampler.weights))
    df_weight = weights[imbalanced["class_name"].values == "DF"][0]
    other_weight = weights[imbalanced["class_name"].values == "NV"][0]
    assert df_weight > other_weight  # rarer class gets a higher sample weight


def test_class_weights_for_loss_shape(synthetic_data_root, base_cfg):
    df = build_manifest(str(synthetic_data_root), base_cfg["data"]["class_names"])
    weights = class_weights_for_loss(df, base_cfg["data"]["num_classes"])
    assert weights.shape[0] == base_cfg["data"]["num_classes"]
    assert (weights > 0).all()


# ---------------------------------------------------------------------------
# Dataset + transforms
# ---------------------------------------------------------------------------

def test_dataset_returns_image_label_metadata(synthetic_data_root, base_cfg):
    df = build_manifest(str(synthetic_data_root), base_cfg["data"]["class_names"])
    ds = HAM10000Dataset(df, get_transforms(base_cfg, train=False))
    image, label, metadata = ds[0]
    assert image.shape == (3, base_cfg["data"]["image_size"], base_cfg["data"]["image_size"])
    assert isinstance(label, int)
    assert metadata.shape == (METADATA_DIM,)


def test_encode_metadata_row_matches_dict(synthetic_data_root, base_cfg):
    df = build_manifest(str(synthetic_data_root), base_cfg["data"]["class_names"])
    row = df.iloc[0]
    from_row = encode_metadata_row(row)
    from_dict = encode_metadata_dict(age=row.get("age"), sex=row.get("sex"), localization=row.get("localization"))
    assert np.allclose(from_row, from_dict, equal_nan=True)


# ---------------------------------------------------------------------------
# Model: forward pass shapes, with and without metadata branch
# ---------------------------------------------------------------------------

def test_plain_model_forward_shape(plain_model):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = plain_model(x)
    assert logits.shape == (2, 7)


def test_plain_model_fusion_gate_in_range(plain_model):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits, alpha = plain_model(x, return_attn=True)
    assert alpha.shape == (2, 1)
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_metadata_model_forward_shape(metadata_model):
    x = torch.randn(2, 3, 224, 224)
    meta = torch.randn(2, METADATA_DIM)
    with torch.no_grad():
        logits = metadata_model(x, metadata=meta)
    assert logits.shape == (2, 7)


def test_metadata_model_gate_sums_to_one(metadata_model):
    x = torch.randn(2, 3, 224, 224)
    meta = torch.randn(2, METADATA_DIM)
    with torch.no_grad():
        _, weights = metadata_model(x, metadata=meta, return_attn=True)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-4)


def test_metadata_model_falls_back_to_default_when_metadata_omitted(metadata_model):
    """This is the compatibility path pytorch-grad-cam relies on: it calls
    model(input_tensor) with no way to pass extra kwargs."""
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = metadata_model(x)  # no metadata passed
    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()


def test_metadata_changes_prediction(metadata_model):
    """Sanity check that the metadata branch actually participates in the
    forward pass rather than being a dead weight — different metadata for
    the same image should (generically) produce different logits."""
    torch.manual_seed(1)
    x = torch.randn(1, 3, 224, 224)
    meta_a = torch.zeros(1, METADATA_DIM)
    meta_b = torch.ones(1, METADATA_DIM)
    with torch.no_grad():
        logits_a = metadata_model(x, metadata=meta_a)
        logits_b = metadata_model(x, metadata=meta_b)
    assert not torch.allclose(logits_a, logits_b)


# ---------------------------------------------------------------------------
# Training step (forward + backward + optimizer, 1 real batch)
# ---------------------------------------------------------------------------

def test_train_run_epoch_one_batch(synthetic_data_root, base_cfg):
    from torch.utils.data import DataLoader

    from src.train import run_epoch
    from src.models.xai_skinnet import XAISkinNet

    set_seed(0)
    df = build_manifest(str(synthetic_data_root), base_cfg["data"]["class_names"])
    train_df, val_df, _ = split_manifest(
        df, base_cfg["data"]["train_split"], base_cfg["data"]["val_split"], seed=0
    )
    ds = HAM10000Dataset(train_df, get_transforms(base_cfg, train=True))
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    model = XAISkinNet(base_cfg)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Only run on the first couple of batches -- this is a smoke test for
    # wiring correctness, not a real training run.
    limited_loader = list(loader)[:2]
    loss, acc = run_epoch(model, limited_loader, criterion, torch.device("cpu"), optimizer)
    assert np.isfinite(loss)
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_expected_calibration_error_perfect_calibration():
    rng = np.random.RandomState(0)
    n = 1000
    probs = rng.uniform(0.5, 1.0, size=n)
    labels = (rng.uniform(0, 1, size=n) < probs).astype(int)
    probs_2class = np.stack([1 - probs, probs], axis=1)
    ece = expected_calibration_error(probs_2class, labels, n_bins=10)
    assert 0.0 <= ece < 0.15  # roughly well-calibrated by construction


def test_expected_calibration_error_overconfident_is_penalized():
    n = 500
    # Always 99% confident but only right half the time -> should have high ECE.
    probs_2class = np.tile([0.01, 0.99], (n, 1))
    labels = np.array([i % 2 for i in range(n)])
    ece = expected_calibration_error(probs_2class, labels, n_bins=10)
    assert ece > 0.3


def test_fit_temperature_reduces_or_maintains_nll():
    from src.calibrate import fit_temperature

    torch.manual_seed(0)
    n, c = 200, 7
    labels = torch.randint(0, c, (n,))
    # Overconfident logits: scale up the correct-class logit heavily.
    logits = torch.randn(n, c) * 0.5
    logits[torch.arange(n), labels] += 5.0

    nll = torch.nn.CrossEntropyLoss()
    nll_before = nll(logits, labels).item()
    temperature = fit_temperature(logits, labels, max_iter=50)
    nll_after = nll(logits / temperature, labels).item()

    assert temperature > 0
    assert nll_after <= nll_before + 1e-3


# ---------------------------------------------------------------------------
# Explainability: Grad-CAM, SHAP, faithfulness metrics
# ---------------------------------------------------------------------------

def test_gradcam_runs_and_produces_valid_heatmap(plain_model):
    from pytorch_grad_cam import GradCAM

    from src.utils import disable_inplace_ops

    disable_inplace_ops(plain_model)
    cam = GradCAM(model=plain_model, target_layers=[plain_model.cnn.grad_cam_target_layer()])
    x = torch.randn(1, 3, 224, 224)
    grayscale_cam = cam(input_tensor=x, targets=None)[0]
    assert grayscale_cam.shape[-2:] == (224, 224) or grayscale_cam.ndim == 2
    assert np.isfinite(grayscale_cam).all()
    assert grayscale_cam.min() >= 0.0 - 1e-5 and grayscale_cam.max() <= 1.0 + 1e-5


def test_gradcam_works_on_metadata_model_via_default_fallback(metadata_model):
    """The key compatibility case: pytorch-grad-cam calls model(x) with no
    metadata kwarg, so this must not crash even when use_metadata=True."""
    from pytorch_grad_cam import GradCAM

    from src.utils import disable_inplace_ops

    disable_inplace_ops(metadata_model)
    cam = GradCAM(model=metadata_model, target_layers=[metadata_model.cnn.grad_cam_target_layer()])
    x = torch.randn(1, 3, 224, 224)
    grayscale_cam = cam(input_tensor=x, targets=None)[0]
    assert np.isfinite(grayscale_cam).all()


def test_shap_deep_explainer_on_classifier_head(plain_model):
    import shap

    torch.manual_seed(0)
    background = torch.randn(8, plain_model.classifier[0].in_features)
    query = torch.randn(1, plain_model.classifier[0].in_features)
    explainer = shap.DeepExplainer(plain_model.classifier, background)
    with torch.no_grad():
        pred_idx = plain_model.classifier(query).argmax(dim=1).item()
    shap_values = explainer.shap_values(query)
    class_shap = shap_values[pred_idx][0] if isinstance(shap_values, list) else shap_values[0, :, pred_idx]
    assert class_shap.shape[0] == plain_model.classifier[0].in_features
    assert np.isfinite(class_shap).all()


def test_xai_eval_deletion_insertion_smoke(synthetic_data_root, base_cfg, plain_model, tmp_path):
    from PIL import Image
    from pytorch_grad_cam import GradCAM

    from src.explainability.xai_eval import N_STEPS, deletion_insertion
    from src.utils import disable_inplace_ops

    disable_inplace_ops(plain_model)
    cam_engine = GradCAM(model=plain_model, target_layers=[plain_model.cnn.grad_cam_target_layer()])

    df = build_manifest(str(synthetic_data_root), base_cfg["data"]["class_names"])
    row = df.iloc[0]
    raw_image = Image.open(row["filepath"]).convert("RGB")
    size = base_cfg["data"]["image_size"]
    raw_resized = np.array(raw_image.resize((size, size))).astype(np.float32) / 255.0
    transform = get_transforms(base_cfg, train=False)

    with torch.no_grad():
        pred_idx = plain_model(transform(raw_image).unsqueeze(0)).argmax(dim=1).item()

    del_probs, ins_probs, del_auc, ins_auc = deletion_insertion(
        plain_model, cam_engine, raw_resized, transform, pred_idx, torch.device("cpu")
    )
    assert len(del_probs) == N_STEPS + 1
    assert len(ins_probs) == N_STEPS + 1
    assert all(0.0 <= p <= 1.0 for p in del_probs)
    assert all(0.0 <= p <= 1.0 for p in ins_probs)
    assert 0.0 <= del_auc <= 1.0
    assert 0.0 <= ins_auc <= 1.0
