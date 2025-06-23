````markdown
# 🧠 Baseline Code for Spatial Reasoning Project

This repository provides baseline implementations for evaluating spatial reasoning using various visual feature extraction and matching methods.

---

## 📦 Environment Setup

To create a conda environment using the provided `environment.yml`:

```bash
conda env create --file environment.yml --name csr
```

---

## 🚀 Running Baselines

### 🔹 NetVLAD

To run NetVLAD-based experiments:

1. Open `main.py`
2. Modify `train_scenes` and `test_scenes` as needed
3. Run the script:

```bash
python main.py
```

---

### 🔹 SuperGlue & RANSAC

For keypoint-based matching using SuperGlue and RANSAC:

- Modify the scene configurations in both `superglue.py` and `ransac.py`
- Run each script separately:

```bash
python superglue.py
python ransac.py
```

---

### 🔹 VGG / ViT / ResNet

To run baselines with different backbone networks:

1. Open `fullscript.py`
2. Set `train_scenes` and `test_scenes` accordingly
3. Update the config section to select a backbone (e.g., VGG, ViT, ResNet) and modify training hyperparameters
4. Run the script:

```bash
python fullscript.py
```

---

### 🔹 SimCLR (Contrastive Learning)

To run the SimCLR-based contrastive learning baseline:

1. Modify `train_scenes` and `test_scenes` in both:
   - `simCLR_kdtree.py`
   - `custom_dataset.py`
2. Run the script:

```bash
python simCLR_kdtree.py
```

---

## 📌 Notes

- All scripts require manual modification of scene configurations before running.
- Outputs and model checkpoints are saved to predefined directories (see in-code comments for details).

---

## 📬 Contact

For questions or contributions, feel free to open an issue or submit a pull request.
