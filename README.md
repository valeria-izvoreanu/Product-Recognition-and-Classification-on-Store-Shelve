# Product-Recognition-and-Classification-on-Store-Shelve

# Grocery Vision Tasks

This repository contains two notebooks that address distinct computer vision challenges using the **GroceryStoreDataset**.

---

## 📦 Task 1: Product Detection on Store Shelves

We develop a system that, given **one reference image per product**, can detect and locate instances of those products within a **store shelf image**.

### ✅ What the system does:
For each product type:
- Detects number of instances
- Calculates the **bounding box dimensions** (width × height in pixels)
- Determines the **position** (center of the bounding box)

### 🖼️ Example Output:
Product 0 - 2 instance found:
  Instance 1 {position: (256, 328), width: 57px, height: 80px}
  Instance 2 {position: (311, 328), width: 57px, height: 80px}


### 🧭 Track A – Single Instance Detection
Detects one instance of each product using feature matching and homography.

### 🔁 Track B – Multiple Instance Detection
Extends Track A to detect multiple instances of the same product in the scene.

---

## 🧠 Task 2: Image Classification on GroceryStoreDataset

We train and evaluate convolutional neural networks to classify grocery products into categories.

### 🛠️ Part 1 – Custom CNN from Scratch
- Designed a CNN using only PyTorch layers (e.g., `nn.Conv2d`, `nn.Linear`).
- No use of prebuilt models like `torchvision.models`.
- Reached 60% validation accuracy.
- Justified all design choices (architecture, training strategy, preprocessing).
- Included results and performance evaluations.

### ⚙️ Part 2 – Fine-tuning ResNet-18
- Fine-tuned a pretrained ResNet-18 (ImageNet-1K) using PyTorch.
- First run used same hyperparameters as custom CNN.
- Second run improved accuracy (reached 87.6% accuracy) by adjusting training strategy.
- Justified improvements based on metrics and training plots.
