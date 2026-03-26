# Head-Wise-Attention-Recalibration-for-ViT-based-STR
Implementation of Head-wise Attention Recalibration (SE-based Head Gating) for Vision Transformer-based Scene Text Recognition


Install the required dependencies using: `pip install -r requirements.txt`



**🔧 How to Run the STR Models**

To train or test the model, run the corresponding script from the folder:
<pre><code>```cd [folder_name] 
./train.ksh   # For training 
./test.ksh    # For testing ```</code></pre>



| **Folder** | **Models** |
|------------|-----------|
| `ViTSTR with Head-wise Attention Recalibration` | `vitstr_tiny_patch16_224_withHeadSE`, `vitstr_small_patch16_224_withHeadSE`, `vitstr_base_patch16_224_withHeadSE` |
| `HTR-VT with Head-wise Attention Recalibration` | `HTR_VT_16_heads_with_headSE`,`HTR_VT_24_heads_with_headSE` |



## **📦 Dataset Preparation**
This project follows the dataset structure and preparation method from the deep-text-recognition-benchmark by CLOVA AI.

**Option 1: Use Preprocessed LMDB Datasets**  
Download ready-to-use LMDB datasets from the CLOVA benchmark:

📂 Directory structure:
<pre><code>``` data/
  └── data_lmdb_release/ 
      ├── training/
      └── evaluation/ ```</code></pre>

- 📎 [Download Links & Details](https://github.com/roatienza/deep-text-recognition-benchmark#download-data)


**Option 2: Create Your Own LMDB Dataset**  
To use your own data, convert it to LMDB format using the `create_lmdb_dataset.py` script from the CLOVA repository:

<pre><code>```bash python3 create_lmdb_dataset.py \
  --input_path path/to/images \
  --gt_file path/to/labels.txt \
  --output_path data_lmdb_release/your_dataset ```</code></pre>

- 🔗 [Full instructions: CLOVA Deep Text Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)


## References
We have used the following works and implementations as the foundation for our models and benchmarking:

- Atienza, Rowel. "Vision transformer for fast and efficient scene text recognition." *International Conference on Document Analysis and Recognition*, pp. 319–334. Springer, 2021.
- Li, Yuting, Dexiong Chen, Tinglong Tang, and Xi Shen. "HTR-VT: Handwritten text recognition with vision transformer." Pattern Recognition 158 (2025): 110967.


