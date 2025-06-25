# Baseline Code for Covision Project

This repository provides baseline implementations for evaluating spatial reasoning using various visual feature extraction and matching methods.

---

## Environment Setup

To create a conda environment using the provided `environment.yml`:

```bash
conda env create --file environment.yml --name covision
```

---

## Running Baselines

### NetVLAD

To run NetVLAD-based experiments:

1. Open `main.py`
2. Modify `train_scenes` and `test_scenes` as needed
3. Run the script:

```bash
python main.py
```

---

### SuperGlue & RANSAC

For keypoint-based matching using SuperGlue and RANSAC:

- Modify the scene configurations in both `superglue.py` and `ransac.py`
- Run each script separately:

```bash
python superglue.py
python ransac.py
```

---

### VGG / ViT / ResNet

To run baselines with different backbone networks:

1. Open `fullscript.py`
2. Set `train_scenes` and `test_scenes` accordingly
3. Update the config section to select a backbone (e.g., VGG, ViT, ResNet) and modify training hyperparameters
4. Run the script:

```bash
python fullscript.py
```

---

### SimCLR (Contrastive Learning)

To run the SimCLR-based contrastive learning baseline:

1. Modify `train_scenes` and `test_scenes` in:
   - `simCLR.py`
2. Run the script:

```bash
python simCLR.py
```

## Notes

- Place the dataset in the project root directory and rename it to `temp`. The expected structure is:

  ```
  ./temp/More_vis/{scene_name_1, scene_name_2, ...}
  ```

- **Scene configurations must be manually specified** before running the scripts. See the in-code comments for guidance. Alternatively, you may use the provided random-split version.

- All output files and model checkpoints will be saved to predefined directories as specified in the scripts.
