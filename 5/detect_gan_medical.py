"""
detect_gan_medical.py

Working implementation of:
"Detection of GAN-manipulated Medical Images through Deep Learning Techniques"

Pipeline:
 - read DICOM/JPG images
 - apply Local Binary Pattern (LBP) preprocessing
 - augment images
 - train U-Net segmentation (L2 regularization)
 - extract encoder features and train SVM classifier
 - evaluate metrics & confusion matrix
 - Grad-CAM heatmaps for explainability

Author: Adapted for user
"""

import os
import glob
import random
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import pydicom
from typing import Tuple, List

# --------------------------
# Dependencies (pip)
# --------------------------
# pip install tensorflow==2.12 numpy opencv-python scikit-learn pydicom joblib matplotlib tqdm

# --------------------------
# Configuration
# --------------------------
IMG_SIZE = 256      # per paper 256x256
BATCH_SIZE = 32
EPOCHS = 25
L2_REG = 1e-4       # kernel_regularizer
LEARNING_RATE = 1e-4
DATA_DIR = "/path/to/dataset"   # <-- change this to your dataset root
# Expect structure:
# DATA_DIR/images/*.png  (or .jpg or .dcm)
# DATA_DIR/masks/*.png   (segmentation masks: class labels per pixel 0..C-1)
# and labels.csv optionally mapping images to class (0:FB,1:FM,2:TB,3:TM)
# If masks not available and only image-level labels exist, adjust to classification flow

# Classes mapping (paper uses 4 classes)
CLASS_NAMES = ['FB', 'FM', 'TB', 'TM']  # false-benign, false-malignant, true-benign, true-malignant

# --------------------------
# Utilities: read image (DICOM/JPG), resize, normalize
# --------------------------
def read_image(path: str, target_size=(IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.dcm']:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        # simple scaling
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Unable to read image: {path}")
    arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_AREA)
    return arr

# --------------------------
# Local Binary Pattern (LBP)
# --------------------------
def lbp_image(gray: np.ndarray, P=8, R=1) -> np.ndarray:
    """
    Compute uniform LBP (Basic) — decimal encoding per pixel.
    returns uint8 LBP image
    """
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.uint8)
    # pad
    for y in range(R, h - R):
        for x in range(R, w - R):
            center = gray[y, x]
            code = 0
            for p in range(P):
                theta = 2.0 * np.pi * p / P
                xp = x + R * np.cos(theta)
                yp = y - R * np.sin(theta)
                # bilinear interpolation
                x1, y1 = int(np.floor(xp)), int(np.floor(yp))
                x2, y2 = int(np.ceil(xp)), int(np.ceil(yp))
                # bounds safe (should be safe due to padding)
                Ia = gray[y1, x1]
                Ib = gray[y1, x2]
                Ic = gray[y2, x1]
                Id = gray[y2, x2]
                # weights
                wx = xp - x1
                wy = yp - y1
                interpolated = (Ia * (1-wx)*(1-wy) + Ib * wx*(1-wy) + Ic*(1-wx)*wy + Id*wx*wy)
                bit = 1 if interpolated >= center else 0
                code |= (bit << p)
            out[y, x] = code
    # scale to 0-255 already uint8
    return out

# --------------------------
# Simple augmentation (tf.image)
# --------------------------
def augment_image(img: np.ndarray) -> np.ndarray:
    # img: grayscale uint8 shape (H,W)
    img_tf = tf.convert_to_tensor(img[..., None], dtype=tf.float32) / 255.0  # [H,W,1]
    # random brightness (paper: gaussian noise elimination [0.5,1.5] - interpret as brightness factor)
    factor = tf.random.uniform([], 0.8, 1.2)  # slight variation
    img_tf = tf.image.adjust_brightness(img_tf, factor - 1.0)
    # random width shift (jitter)
    # translate horizontally by up to 10% of width
    w = tf.shape(img_tf)[1]
    max_dx = tf.cast(0.1 * tf.cast(w, tf.float32), tf.int32)
    dx = tf.random.uniform([], -max_dx, max_dx + 1, dtype=tf.int32)
    img_tf = tfa.image.translate(img_tf, translations=[dx, 0]) if 'tfa' in globals() else tf.roll(img_tf, shift=dx, axis=1)
    # random horizontal flip
    img_tf = tf.image.random_flip_left_right(img_tf)
    # gaussian noise
    noise = tf.random.normal(tf.shape(img_tf), mean=0.0, stddev=0.01)
    img_tf = img_tf + noise
    img_tf = tf.clip_by_value(img_tf, 0.0, 1.0)
    out = tf.cast(img_tf * 255.0, tf.uint8).numpy()[..., 0]
    return out

# --------------------------
# Dataset generator
# --------------------------
def build_dataset(image_paths: List[str], mask_paths: List[str], labels: List[int],
                  batch_size=BATCH_SIZE, shuffle=True, augment=False):
    """Builds tf.data Dataset yielding (image, mask, label) where mask may be None."""
    def gen():
        for img_p, mask_p, label in zip(image_paths, mask_paths, labels):
            img = read_image(img_p)
            # LBP preprocess (per-paper)
            lbp = lbp_image(img)
            # stack original + lbp as channels (paper used LBP preprocessing before UNet; it's ambiguous whether they used single-channel LBP or combined — here we stack)
            inp = np.stack([img, lbp], axis=-1).astype(np.float32) / 255.0  # shape H W 2
            mask = None
            if mask_p is not None:
                m = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
                if m is None: m = np.zeros((IMG_SIZE,IMG_SIZE), dtype=np.uint8)
                m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), cv2.INTER_NEAREST)
                mask = m.astype(np.int32)
            yield inp, (mask if mask is not None else -1), label

    output_types = (tf.float32, tf.int32, tf.int32)
    output_shapes = (tf.TensorShape([IMG_SIZE, IMG_SIZE, 2]),
                     tf.TensorShape([IMG_SIZE, IMG_SIZE]),
                     tf.TensorShape([]))
    ds = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
    if shuffle:
        ds = ds.shuffle(buffer_size=1024)
    def _map(inp, mask, lbl):
        if augment:
            # apply augmentation on first channel (original) and LBP channel separately? We'll apply same transform to both for alignment.
            img_uint8 = tf.cast(inp[...,0]*255.0, tf.uint8)
            lbp_uint8 = tf.cast(inp[...,1]*255.0, tf.uint8)
            # use numpy augmentation for reliability (slow). For production, write tf augmenters.
            inp_np = tf.numpy_function(func=lambda a,b: np.stack([augment_image(a), augment_image(b)], axis=-1),
                                       inp=[img_uint8, lbp_uint8],
                                       Tout=tf.uint8)
            inp_np = tf.cast(inp_np, tf.float32) / 255.0
            return inp_np, mask, lbl
        else:
            return inp, mask, lbl
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# --------------------------
# U-Net model builder (Keras) with L2 regularization
# --------------------------
def conv_block(x, filters, l2_reg=L2_REG):
    x = layers.Conv2D(filters, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(filters, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    return x

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 2), num_classes=4, l2_reg=L2_REG):
    inputs = layers.Input(shape=input_shape)
    # encoder
    c1 = conv_block(inputs, 64, l2_reg); p1 = layers.MaxPooling2D((2,2))(c1)
    c2 = conv_block(p1, 128, l2_reg); p2 = layers.MaxPooling2D((2,2))(c2)
    c3 = conv_block(p2, 256, l2_reg); p3 = layers.MaxPooling2D((2,2))(c3)
    c4 = conv_block(p3, 512, l2_reg); p4 = layers.MaxPooling2D((2,2))(c4)
    c5 = conv_block(p4, 1024, l2_reg)
    # decoder
    u6 = layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4]); c6 = conv_block(u6, 512, l2_reg)
    u7 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3]); c7 = conv_block(u7, 256, l2_reg)
    u8 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2]); c8 = conv_block(u8, 128, l2_reg)
    u9 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1]); c9 = conv_block(u9, 64, l2_reg)
    outputs = layers.Conv2D(num_classes, (1,1), activation='softmax')(c9)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    # Also create encoder model to extract features
    encoder_model = models.Model(inputs, c5)  # bottleneck output
    return model, encoder_model

# --------------------------
# Grad-CAM helper (for segmentation/class maps)
# --------------------------
def make_gradcam_heatmap(model: tf.keras.Model, img_tensor: np.ndarray, class_index:int, last_conv_layer_name=None):
    """
    img_tensor: shape (1, H, W, C)
    class_index: channel index in model output (for segmentation classes)
    """
    if last_conv_layer_name is None:
        # attempt to find last conv
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor)
        loss = preds[..., class_index]  # shape (1,H,W)
    grads = tape.gradient(loss, conv_outputs)  # shape (1,H,W,filters)
    # global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(1,2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads[0].numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[..., i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    return heatmap

# --------------------------
# Training and evaluation pipeline
# --------------------------
def load_dataset_filenames_and_labels(data_dir: str):
    """
    Placeholder loader: expects:
     - data_dir/images/*.png or .dcm
     - data_dir/masks/*.png  (or none)
     - data_dir/labels.csv optional mapping of filename -> label
    If mask missing, mask_paths filled with None
    labels: must be mapped to 0..num_classes-1
    """
    imgs = sorted(glob.glob(os.path.join(data_dir, "images", "*")))
    masks = sorted(glob.glob(os.path.join(data_dir, "masks", "*"))) if os.path.isdir(os.path.join(data_dir, "masks")) else []
    # naive matching by filename
    mask_map = {os.path.basename(p): p for p in masks}
    # labels: if labels.csv exists, load mapping, else detect from filename with pattern containing class name
    label_map = {}
    labels_csv = os.path.join(data_dir, "labels.csv")
    if os.path.exists(labels_csv):
        import csv
        with open(labels_csv, 'r') as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row: continue
                fname, lbl = row[0].strip(), row[1].strip()
                label_map[fname] = lbl
    image_paths, mask_paths, labels = [], [], []
    le = LabelEncoder()
    # attempt to build labels from filename token
    temp_labels = []
    for p in imgs:
        fn = os.path.basename(p)
        if fn in label_map:
            temp_labels.append(label_map[fn])
        else:
            # try to infer class token like _FB_ or FB in name
            assigned = None
            for cname in CLASS_NAMES:
                if cname.lower() in fn.lower():
                    assigned = cname
                    break
            if assigned is None:
                # fallback: assume 'Unaltered' if 'true' found -> TB/TM not distinguished
                assigned = 'TB'  # default
            temp_labels.append(assigned)
    le.fit(CLASS_NAMES)
    for p, lbl in zip(imgs, temp_labels):
        image_paths.append(p)
        fname = os.path.basename(p)
        mask_paths.append(mask_map.get(fname, None))
        labels.append(int(le.transform([lbl])[0]))
    return image_paths, mask_paths, labels

def train_and_evaluate():
    image_paths, mask_paths, labels = load_dataset_filenames_and_labels(DATA_DIR)
    # split
    combined = list(zip(image_paths, mask_paths, labels))
    random.shuffle(combined)
    n = len(combined)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    train = combined[:train_end]
    val = combined[train_end:val_end]
    test = combined[val_end:]
    train_imgs, train_masks, train_lbls = zip(*train)
    val_imgs, val_masks, val_lbls = zip(*val)
    test_imgs, test_masks, test_lbls = zip(*test)
    print(f"Counts: train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

    train_ds = build_dataset(list(train_imgs), list(train_masks), list(train_lbls), augment=True, shuffle=True)
    val_ds = build_dataset(list(val_imgs), list(val_masks), list(val_lbls), augment=False, shuffle=False)

    # build model
    num_classes = len(CLASS_NAMES)
    unet, encoder = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 2), num_classes=num_classes, l2_reg=L2_REG)
    unet.compile(optimizer=Adam(LEARNING_RATE),
                 loss=SparseCategoricalCrossentropy(),
                 metrics=['accuracy'])
    unet.summary()

    # callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("unet_best.h5", save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    # train
    unet.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

    # load best
    unet = tf.keras.models.load_model("unet_best.h5", compile=False)
    # For encoder, rebuild using loaded weights: create same architecture and set weights
    _, encoder = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 2), num_classes=num_classes, l2_reg=L2_REG)
    # copy weights for matching layers
    # naive approach: map layers by name if needed -> here we'll just re-load weights into the whole model and then extract c5 output
    full_model, enc_model_temp = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 2), num_classes=num_classes, l2_reg=L2_REG)
    full_model.load_weights("unet_best.h5")
    encoder = models.Model(full_model.input, full_model.get_layer(index=10).output)  # index may vary based on construction; safer: find bottleneck by shape
    # To be safe, find layer with largest channel dims as bottleneck
    bottleneck_layer = None
    max_filters = 0
    for layer in full_model.layers:
        if isinstance(layer, layers.Conv2D):
            if layer.output_shape and layer.output_shape[-1] and layer.output_shape[-1] > max_filters:
                max_filters = layer.output_shape[-1]
                bottleneck_layer = layer
    if bottleneck_layer is None:
        raise RuntimeError("Cannot find bottleneck layer automatically")
    encoder = models.Model(full_model.input, bottleneck_layer.output)

    # ---------------------------
    # Extract features for SVM
    # ---------------------------
    def extract_features_for_list(paths, mask_paths, labels_list):
        feats = []
        labs = []
        for p, m, lbl in tqdm(zip(paths, mask_paths, labels_list), total=len(paths), desc="Extract feats"):
            img = read_image(p)
            lbp = lbp_image(img)
            inp = np.stack([img, lbp], axis=-1).astype(np.float32)/255.0
            inp_batch = np.expand_dims(inp, axis=0)
            feat_map = encoder.predict(inp_batch)  # shape (1,Hf,Wf,channels)
            feat_vec = feat_map.reshape(-1)  # flatten
            feats.append(feat_vec)
            labs.append(lbl)
        feats = np.stack(feats, axis=0)
        labs = np.array(labs)
        return feats, labs

    # extract for train/val/test
    tr_feats, tr_labels = extract_features_for_list(train_imgs, train_masks, train_lbls)
    val_feats, val_labels = extract_features_for_list(val_imgs, val_masks, val_lbls)
    test_feats, test_labels = extract_features_for_list(test_imgs, test_masks, test_lbls)

    # scale
    scaler = StandardScaler()
    tr_feats_s = scaler.fit_transform(tr_feats)
    val_feats_s = scaler.transform(val_feats)
    test_feats_s = scaler.transform(test_feats)
    joblib.dump(scaler, "scaler.joblib")

    # Train SVM
    print("Training SVM classifier on encoder features...")
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(tr_feats_s, tr_labels)
    joblib.dump(svm, "svm_model.joblib")

    # Evaluate SVM
    pred_test = svm.predict(test_feats_s)
    pred_prob = svm.predict_proba(test_feats_s) if hasattr(svm, "predict_proba") else None
    print("Classification report (SVM on encoder features):")
    print(classification_report(test_labels, pred_test, target_names=CLASS_NAMES))
    cm = confusion_matrix(test_labels, pred_test)
    print("Confusion matrix:\n", cm)
    if pred_prob is not None:
        try:
            auc = roc_auc_score(tf.keras.utils.to_categorical(test_labels), pred_prob, multi_class='ovr')
            print("ROC AUC (ovr):", auc)
        except Exception as e:
            print("ROC AUC failed:", e)

    # Also evaluate segmentation qualitatively & compute per-pixel accuracy on test set (optional)
    # produce sample Grad-CAM visualizations
    sample_idx = np.random.choice(len(test_imgs), min(8, len(test_imgs)), replace=False)
    for idx in sample_idx:
        p = test_imgs[idx]
        img = read_image(p)
        lbp = lbp_image(img)
        inp = np.stack([img, lbp], axis=-1).astype(np.float32)/255.0
        inp_batch = np.expand_dims(inp, axis=0)
        seg_pred = unet.predict(inp_batch)[0]  # shape (H,W,num_classes)
        class_map = np.argmax(seg_pred, axis=-1).astype(np.uint8)
        # convert class_map to color overlay
        heatmap = make_gradcam_heatmap(unet, inp_batch, class_index=np.argmax(np.bincount(class_map.flatten())))
        overlay = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
        gray_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(gray_rgb, 0.6, overlay, 0.4, 0)
        outp = os.path.join("gradcam_samples", f"sample_{idx}.png")
        os.makedirs("gradcam_samples", exist_ok=True)
        cv2.imwrite(outp, blended)
    print("Grad-CAM samples saved to gradcam_samples/")

if __name__ == "__main__":
    train_and_evaluate()
