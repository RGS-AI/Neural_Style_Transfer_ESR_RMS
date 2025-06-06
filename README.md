# Neural Style Transfer with High-Resolution Upscaling (RRDB, PyTorch)

This project applies **Neural Style Transfer (NST)** to input images and then enhances the resolution using **Real-ESRGAN RRDB models** (PyTorch-based). The final output is a high-quality, stylized image suitable for large-format use.

***

## 🚀 Key Features

- Consistent **style transfer** using a reference style image.
- High-quality **image upscaling** using Real-ESRGAN RRDB model (realesrgan-x4plus.pth).
- Complete **PyTorch-based pipeline** — no Vulkan binaries required.
- Compatible with **macOS (CPU)** and Linux.

***

## 📁 Project Structure
<pre>
nst_upscale_env/
│
├── style_transfer/
│   ├── nst_model.py         # PyTorch code for NST
│   ├── utils.py             # Preprocessing and helpers
│   ├── content.jpg          # Original input image
│   ├── style.jpg            # Artistic style reference
│   └── stylized_output.jpg  # NST output
│
├── upscaling/
│   ├── realesrgan/                  # Cloned Real-ESRGAN repo (PyTorch)
│   ├── test.py                      # Script to run the upscaler
│   └── weights/
│       └── RealESRGAN_x4plus.pth    # Pretrained RRDB model
│
├── requirements.txt
├── README.md
└── .gitignore
</pre>

***

## 🧱 Setup Instructions

### 1. Create and Activate Virtual Environment

`python3 -m venv nst_upscale_env`

`source nst_upscale_env/bin/activate`

### 2. Clone This Repository

`git clone https://github.com/<your-username>/NST_Upscale.git`

`cd NST_Upscale`

### 3. Install Dependencies

`pip install -r requirements.txt`

### 🎨 Step 1: Run Neural Style Transfer

`cd style_transfer`

`python NST.ipynb`  #nst_model.py

• This script will read content.jpg and style.jpg, then save the output as stylized_output.jpg.

### 🔼 Step 2: Upscale the Stylized Image

**Download Weights (if not already):**

**Download the RealESRGAN_x4plus.pth model from the official repo:**

`https://github.com/xinntao/Real-ESRGAN#model-zoo`

**Place it inside:**

`upscaling/realesrgan/weights/`

**Run Upscaler:**

`cd ESRGAN`

`python3 test.py`

• Output image will be saved as output in .png

***

## 🖼️ Screenshots

### Input

**Content Image**  
![Content](screenshots/landscape.jpg)

**Style Image**  
![Style](screenshots/landscape_Style.jpg)

### Output

**Stylized Output (512x512)**  
![Stylized Output](screenshots/stylized_output.jpg)

**Upscaled to High Resolution**  
![Upscaled Output](screenshots/stylized_output_rlt.png)

***

**⚠️ Notes**<br>

• Ensure content and style images are of compatible sizes. The script resizes them automatically to 512x512 before NST<br>
• Real-ESRGAN uses the RRDB-based ESRGAN model for super-resolution<br>
• PyTorch CPU-only execution can be slower. GPU acceleration requires CUDA (not available on macOS by default)<br>

***

**📘 References**<br>

• Neural Style Transfer — PyTorch Tutorial<br>
• Real-ESRGAN Repository - Credits of ESRGAN to xintaowang

***

**👤 Author**

Raghunandan M S<br>
Data Scientist | AI Enthusiast<br>
GitHub: github.com/RGS-AI
