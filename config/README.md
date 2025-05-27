# Spotscape
```yaml
dataset: 
    # Single-slice SDI 
    10X_DLPFC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.0001
        lam_re: 0.1 # Balancing parameter for Reconstruction loss
        lam_sc: 1.0 # Balancing parameter for Similarity Telescope (SC) Loss
    MTG:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.0001
        lam_re: 0.1 # Balancing parameter for Reconstruction loss
        lam_sc: 1.0 # Balancing parameter for Similarity Telescope (SC) Loss
    Mouse_Embryo:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.00001
        lam_re: 0.1 # Balancing parameter for Reconstruction loss
        lam_sc: 1.0 # Balancing parameter for Similarity Telescope (SC) Loss
    NSCLC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.0005
        lam_re: 0.1 # Balancing parameter for Reconstruction loss
        lam_sc: 1.0 # Balancing parameter for Similarity Telescope (SC) Loss
    # Multi-slice Homogeneous Integration
    10X_DLPFC:
        patient_idx: 0
        slice_idx: -1
        epochs: 1000
        lr: 0.0005
        lam_re: 0.1 # Balancing parameter for Reconstruction loss
        lam_sc: 1.0 # Balancing parameter for Similarity Telescope (SC) Loss
        lam_pcl: 0.01 # Balancing parameter for Prototypical Contrastive Loss
        lam_ss: 1.0 # Balancing parameter for Similarity Scaling (SS) Loss
    MTG:
        patient_idx: 0
        slice_idx: -1
        epochs: 1000
        lr: 0.0005
        lam_re: 0.1 # Balancing parameter for Reconstruction loss
        lam_sc: 1.0 # Balancing parameter for Similarity Telescope (SC) Loss
        lam_pcl: 0.01 # Balancing parameter for Prototypical Contrastive Loss
        lam_ss: 1.0 # Balancing parameter for Similarity Scaling (SS) Loss
    # Alignment
    Mouse_Embryo:
        patient_idx: 0
        slice_idx: -1
        epochs: 1000
        lr: 0.001
        lam_re: 0.1 # Balancing parameter for Reconstruction loss
        lam_sc: 1.0 # Balancing parameter for Similarity Telescope (SC) Loss
        lam_pcl: 0.01 # Balancing parameter for Prototypical Contrastive Loss
        lam_ss: 1.0 # Balancing parameter for Similarity Scaling (SS) Loss
    Breast_Cancer: # Visium-Xenium
        patient_idx: 0
        slice_idx: -1
        epochs: 1000
        lr: 0.00001
        lam_re: 0.1 # Balancing parameter for Reconstruction loss
        lam_sc: 1.0 # Balancing parameter for Similarity Telescope (SC) Loss
        lam_pcl: 0.01 # Balancing parameter for Prototypical Contrastive Loss
        lam_ss: 1.0 # Balancing parameter for Similarity Scaling (SS) Loss
```

# SEDR

```yaml
dataset:
  # Single-slice SDI 
  10X_DLPFC:
    patient_idx: 0
    slice_idx: 0
    epochs: 200
    lr: 0.0001
    rec_w: 10.0 # balancing parameter for reconstruction loss
    gcn_w: 0.1 # balancing parameter for gcn loss
    self_w: 1.0 # balancing parameter for self loss
  MTG:
    patient_idx: 0
    slice_idx: 0
    epochs: 200
    lr: 0.0001
    rec_w: 10.0 # balancing parameter for reconstruction loss
    gcn_w: 0.1 # balancing parameter for gcn loss
    self_w: 0.1 # balancing parameter for self loss
  Mouse_Embryo:
    patient_idx: 0
    slice_idx: 0
    epochs: 1000
    lr: 1e-05
    rec_w: 10.0 # balancing parameter for reconstruction loss
    gcn_w: 0.1 # balancing parameter for gcn loss
    self_w: 0.1 # balancing parameter for self loss
  NSCLC:
    patient_idx: 0
    slice_idx: 0
    epochs: 1000
    lr: 0.0001
    rec_w: 10.0 # balancing parameter for reconstruction loss
    gcn_w: 0.1 # balancing parameter for gcn loss
    self_w: 1.0 # balancing parameter for self loss
# Multi-slice Homogeneous Integration
  10X_DLPFC:
    patient_idx: 0
    slice_idx: -1
    epochs: 200
    lr: 0.0001
    rec_w: 10.0 # balancing parameter for reconstruction loss
    gcn_w: 0.1 # balancing parameter for gcn loss
    self_w: 1.0 # balancing parameter for self loss
```

# STAGATE

```yaml
dataset:
    # Single-slice SDI 
    10X_DLPFC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.005
    MTG:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.0005
    Mouse_Embryo:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.0001
    NSCLC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.005
    # Multi-slice Homogeneous Integration
    10X_DLPFC:
        patient_idx: 0
        slice_idx: -1
        epochs: 1000
        lr: 0.005
```

# SpaCAE

```yaml
dataset: 
    # Single-slice SDI 
    10X_DLPFC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.05
        alpha: 1.0 # spatial expression augmentation parameter
    MTG:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.05
        alpha: 0.5 # spatial expression augmentation parameter
    Mouse_Embryo:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 5e-05
        alpha: 0.5 # spatial expression augmentation parameter
    NSCLC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.001
        alpha: 1.0 # spatial expression augmentation parameter
    # Multi-slice Homogeneous Integration 
    10X_DLPFC:
        patient_idx: 0
        slice_idx: -1
        epochs: 1000
        lr: 0.001
        alpha: 1.0 # spatial expression augmentation parameter
```

# Spaceflow

```yaml
dataset: 
    # Single-slice SDI 
    10X_DLPFC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.0005
        lam_reg: 0.1 # spatial consistency loss balancing parameter
    MTG:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.0001
        lam_reg: 1.0 # spatial consistency loss balancing parameter
    Mouse_Embryo:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.001
        lam_reg: 0.1 # spatial consistency loss balancing parameter
    NSCLC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.0001
        lam_reg: 0.1 # spatial consistency loss balancing parameter
    # Multi-slice Homogeneous Integration
    10X_DLPFC:
        patient_idx: 0
        slice_idx: -1
        epochs: 1000
        lr: 0.0001
        lam_reg: 0.1 # spatial consistency loss balancing parameter
```


# GraphST

```yaml
dataset:
    # Single-slice SDI 
    10X_DLPFC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.0005
        alpha: 1.0 # balancing parameter for feature reconstruction
        beta: 0.1 # balancing parameter for self-supervised contrastive loss
    MTG:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.001
        alpha: 10.0 # balancing parameter for feature reconstruction
        beta: 10.0 # balancing parameter for self-supervised contrastive loss
    Mouse_Embryo:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.005
        alpha: 10.0 # balancing parameter for feature reconstruction
        beta: 1.0 # balancing parameter for self-supervised contrastive loss
    NSCLC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 0.0005
        alpha: 1.0 # balancing parameter for feature reconstruction
        beta: 0.1 # balancing parameter for self-supervised contrastive loss
    dataset:
    # Multi-slice Homogeneous Integration
    10X_DLPFC:
        patient_idx: 0
        slice_idx: -1
        epochs: 1000
        lr: 0.005
        alpha: 1.0 # balancing parameter for feature reconstruction
        beta: 1.0 # balancing parameter for self-supervised contrastive loss
    # Multi-slice Heterogeneous Integration
    MTG:
        patient_idx: 0
        slice_idx: -1
        epochs: 1000
        lr: 0.001
        alpha: 10.0 # balancing parameter for feature reconstruction
        beta: 10.0 # balancing parameter for self-supervised contrastive loss
```

# MAFN

```yaml
dataset:
    # Single-slice SDI 
    10X_DLPFC:
        patient_idx: 0
        slice_idx: 0
        epochs: 250
        lr: 0.005
        alpha: 10.0 # Balancing parameter for ZINB loss
        beta: 1.0 # Balancing parameter for Regularization loss
    MTG:
        patient_idx: 0
        slice_idx: 0
        epochs: 250
        lr: 0.0005
        alpha: 10.0 # Balancing parameter for ZINB loss
        beta: 10.0 # Balancing parameter for Regularization loss
    # Mouse_Embryo: # OOM
    #   patient_idx: 0
    #   slice_idx: 0
    NSCLC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1000
        lr: 5e-05
        alpha: 0.1 # Balancing parameter for ZINB loss
        beta: 0.1 # Balancing parameter for Regularization loss
```

# stGCL

```yaml
dataset:
    # Single-slice SDI 
    10X_DLPFC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1200
        lr: 0.005
        alpha: 0.5 # Balancing parameter for DGI (self-supervised) loss
    MTG:
        patient_idx: 0
        slice_idx: 0
        epochs: 1200
        lr: 0.001
        alpha: 0.5 # Balancing parameter for DGI (self-supervised) loss
    Mouse_Embryo:
        patient_idx: 0
        slice_idx: 0
        epochs: 1200
        lr: 0.005
        alpha: 0.1 # Balancing parameter for DGI (self-supervised) loss
    NSCLC:
        patient_idx: 0
        slice_idx: 0
        epochs: 1200
        lr: 0.005
        alpha: 5.0 # Balancing parameter for DGI (self-supervised) loss
    # Multi-slice Homogeneous Integration
    10X_DLPFC:
        patient_idx: 0
        slice_idx: -1
        epochs: 1200
        lr: 0.005
        alpha: 0.5 # Balancing parameter for DGI (self-supervised) loss
```

# STAligner

```yaml
dataset:
    # Multi-slice Homogeneous Integration
    10X_DLPFC:
        patient_idx: 0
        slice_idx: -1
        epochs_stagate: 1000
        lr_stagate: 0.005
        epochs: 1000
        lr: 0.01
    # Multi-slice Heterogeneous Integration
    MTG:
        patient_idx: 0
        slice_idx: -1
        epochs_stagate: 1000
        lr_stagate: 0.005
        epochs: 1000
        lr: 0.005
```

# CAST

```yaml
dataset:
    # Multi-slice Homogeneous Integration
    10X_DLPFC:
        patient_idx: 0
        slice_idx: -1
        epochs: 400
        lr: 1e-05
        lambd: 0.001 # Balancing parameter for CCA-SSG (self-supervised) loss
    # Multi-slice Heterogeneous Integration
    MTG:
        patient_idx: 0
        slice_idx: -1
        epochs: 400
        lr: 5e-05
        lambd: 0.001 # Balancing parameter for CCA-SSG (self-supervised) loss
```