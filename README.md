# Neural Style Transfer

A deep learning-based **Neural Style Transfer (NST)** application that combines the content of one image with the artistic style of another image.

The project uses **Adaptive Instance Normalization (AdaIN)** to perform fast arbitrary style transfer and provides a **Flask web interface** where users can upload their own content and style images and generate a stylized result.

---

## Project Overview

Neural Style Transfer is a computer vision technique that allows the visual style of one image to be transferred onto another image while preserving the main structure and content of the original image.

For example:

**Content Image:** a photograph of a person or scene

**Style Image:** a painting or artwork

**Output:** the original content represented using the visual characteristics of the style image.

This project implements the AdaIN-based approach proposed in:

> **Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization**
> Huang & Belongie, ICCV 2017

Unlike traditional style transfer approaches that optimize an image separately for each content-style pair, AdaIN allows arbitrary content and style images to be processed using a trained encoder-decoder network.

---

## Demo

The project includes a Flask-based web application where users can:

* Upload a content image
* Upload a style image
* Control the style-transfer strength using an alpha value
* Generate a stylized image
* View the input and generated images
* Explore example content/style combinations
* Learn about the project through the application's FAQ section

### Example

| Content Image      | Style Image       | Stylized Result                         |
| ------------------ | ----------------- | --------------------------------------- |
| Content photograph | Artistic painting | Content with transferred artistic style |

---

## How Neural Style Transfer Works

The overall pipeline is:

```text
Content Image
      │
      ▼
Resize & Preprocess
      │
      ▼
VGG-19 Encoder
      │
      ▼
Content Feature Representation
      │
      │
      │              Style Image
      │                   │
      │                   ▼
      │             Resize & Preprocess
      │                   │
      │                   ▼
      │              VGG-19 Encoder
      │                   │
      │                   ▼
      │             Style Features
      │                   │
      └──────────┬────────┘
                 ▼
       Adaptive Instance
          Normalization
                 │
                 ▼
       Stylized Feature Map
                 │
                 ▼
              Decoder
                 │
                 ▼
         Stylized Image
```

The important idea is that the model does not directly combine the pixels of the two images.

Instead, it works in **feature space**.

---

# Core Concept: AdaIN

The main technique used in this project is **Adaptive Instance Normalization (AdaIN)**.

The content and style images are first passed through a pretrained VGG encoder to obtain feature representations.

AdaIN aligns the channel-wise mean and variance of the content features with those of the style features.

The basic operation can be represented as:

```text
AdaIN(content, style)
=
style_mean
+
style_std × normalized_content
```

More formally:

```text
AdaIN(x, y) =
σ(y) * ((x - μ(x)) / σ(x)) + μ(y)
```

Where:

* `x` = content feature map
* `y` = style feature map
* `μ(x)` = channel-wise mean of content features
* `σ(x)` = channel-wise standard deviation of content features
* `μ(y)` = channel-wise mean of style features
* `σ(y)` = channel-wise standard deviation of style features

This allows the statistical characteristics of the style features to be transferred to the content features.

---

# Why VGG?

The project uses a pretrained **VGG-19 network** as the feature encoder.

VGG is not used here primarily as an image classifier.

Instead, intermediate convolutional layers are used to extract meaningful visual features.

These features contain information about:

* Shapes
* Edges
* Textures
* Patterns
* Object structures
* Higher-level visual information

The encoder therefore converts the input image from pixel space into a representation that is useful for style transfer.

---

# Encoder

The project contains a custom `VGGEncoder`.

Its main responsibilities are:

1. Load the pretrained VGG weights.
2. Extract intermediate feature representations.
3. Convert input images into feature-space representations.
4. Provide those features to the AdaIN operation.

The encoder is used for both:

```text
Content Image → Content Features
```

and

```text
Style Image → Style Features
```

The encoder parameters remain fixed during inference.

---

# Adaptive Instance Normalization

After extracting the two feature representations:

```text
Content Features
        +
Style Features
        ↓
      AdaIN
        ↓
Stylized Features
```

AdaIN modifies the statistics of the content feature representation so that they match the statistics of the style representation.

This is what enables the model to work with **arbitrary style images** instead of being restricted to a predefined set of styles.

---

# Decoder

After AdaIN produces the stylized feature representation, the decoder converts those features back into image space.

```text
Stylized Features
       ↓
    Decoder
       ↓
Stylized Image
```

The decoder is trained to approximately reverse the transformation performed by the encoder.

Therefore, the complete process is:

```text
Image
 ↓
VGG Encoder
 ↓
Feature Representation
 ↓
AdaIN
 ↓
Stylized Feature Representation
 ↓
Decoder
 ↓
Output Image
```

---

# Alpha Parameter

The application includes an `alpha` parameter that controls the strength of the style transfer.

The interpolation is:

```text
output_features =
alpha × stylized_features
+
(1 - alpha) × content_features
```

### When alpha = 1

The result contains the maximum style transformation.

```text
alpha = 1.0
```

### When alpha = 0

The original content features are retained.

```text
alpha = 0.0
```

### Intermediate values

For example:

```text
alpha = 0.5
```

produces a mixture between the original content representation and the stylized representation.

This provides users with control over how strongly the artistic style affects the final image.

---

# Image Preprocessing

Before inference, both content and style images are:

1. Loaded using Pillow.
2. Converted to RGB.
3. Resized to `256 × 256`.
4. Converted into PyTorch tensors.
5. A batch dimension is added.
6. The tensors are moved to the available device.

The pipeline is approximately:

```text
Image
 ↓
PIL.Image
 ↓
RGB Conversion
 ↓
Resize(256 × 256)
 ↓
ToTensor()
 ↓
Add Batch Dimension
 ↓
PyTorch Tensor
```

---

# Inference Pipeline

The actual inference process follows these steps:

### Step 1 — Load images

The Flask application receives the uploaded content and style images.

### Step 2 — Preprocess

Both images are resized and converted into tensors.

### Step 3 — Feature extraction

The VGG encoder extracts:

```text
content_features
style_features
```

### Step 4 — Apply AdaIN

The content features are transformed using the style statistics.

### Step 5 — Apply alpha blending

The stylized features are blended with the original content features according to the selected alpha value.

### Step 6 — Decode

The decoder reconstructs the final image.

### Step 7 — Save output

The generated tensor is converted back into a PIL image and saved.

### Step 8 — Display

Flask sends the generated image back to the web interface.

---

# Web Application

The project uses **Flask** as the backend framework.

The Flask application handles:

* File uploads
* Input validation
* Image preprocessing
* Model inference
* Output generation
* Serving uploaded images
* Serving example images
* Rendering the HTML interface

The main application flow is:

```text
User
 ↓
Flask Web Interface
 ↓
Upload Content + Style
 ↓
Validate Files
 ↓
Save Images
 ↓
Preprocess Images
 ↓
Run NST Model
 ↓
Save Generated Image
 ↓
Display Result
```

---

# Project Structure

```text
Neural-Style-Transfer/
│
├── app.py
│
├── decoder_12.pth
│
├── requirements.txt
├── Procfile
├── .python-version
├── .gitignore
│
├── templates/
│   ├── index.html
│
├── static/
│   └── uploads/
│
├── examples/
│   ├── Jimin.jpg
│   ├── la_muse.jpg
│   ├── stylized_jimin_la_muse.jpg
│   ├── Jungkook.jpg
│   ├── The_Scream.jpg
│   └── stylized_Jungkook_the_scream.jpg
│
├── utils/
│   ├── models.py
│   └── utils.py
│
└── vgg/
    └── vgg_normalised.pth
```

---

# Important Files

## `app.py`

The main Flask application.

It contains:

* Flask configuration
* Upload form
* Model loading
* Image preprocessing
* Style transfer function
* Image saving
* Flask routes

The central inference function performs:

```text
Content Image
        ↓
Content Features
        ↓
AdaIN ← Style Features
        ↓
Stylized Features
        ↓
Decoder
        ↓
Stylized Image
```

---

## `utils/models.py`

Contains the model architecture used by the project.

This includes the:

* VGG encoder
* Decoder

The encoder extracts feature representations while the decoder reconstructs images from those representations.

---

## `utils/utils.py`

Contains utility functions required by the model.

One of the important functions is:

```python
adaptive_instance_normalization()
```

which performs the AdaIN operation.

---

## `decoder_12.pth`

Contains the trained decoder weights.

The decoder uses these learned parameters to reconstruct an image from the stylized feature representation.

---

## `vgg/vgg_normalised.pth`

Contains the pretrained VGG weights used by the encoder.

The VGG network acts as the feature extraction component of the style-transfer pipeline.

---

# Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* Torchvision
* VGG-19
* Adaptive Instance Normalization

### Computer Vision

* Pillow
* Image preprocessing
* Feature extraction

### Web Development

* Flask
* Flask-WTF
* Flask-Bootstrap
* HTML
* CSS

### Deployment

* Render

---

# Installation

Clone the repository:

```bash
git clone https://github.com/AlishaShaikh20/Neural-Style-Transfer.git
```

Move into the project directory:

```bash
cd Neural-Style-Transfer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start the Flask application:

```bash
python app.py
```

The application will run locally at:

```text
http://127.0.0.1:5000
```

Open the address in a browser.

Then:

1. Upload a content image.
2. Upload a style image.
3. Select the desired alpha value.
4. Click **Transfer Style**.
5. View the generated image.

---

# Requirements

The project requires:

```text
Flask
Flask-Bootstrap
Flask-WTF
Pillow
PyTorch
Torchvision
tqdm
Werkzeug
WTForms
Gunicorn
```

Exact versions used for deployment are available in:

```text
requirements.txt
```

---

# Example Results

The repository includes example content/style pairs demonstrating the output of the model.

### Example 1

```text
Content:
Jimin photograph

Style:
La Muse

Result:
Jimin image with La Muse-inspired artistic style
```

### Example 2

```text
Content:
Jungkook photograph

Style:
The Scream

Result:
Jungkook image with The Scream-inspired artistic style
```

These examples demonstrate that the model can apply different artistic styles to different content images.

---

# Training and Inference

The project uses a pretrained VGG network for feature extraction and a trained decoder for image reconstruction.

During inference, the encoder and decoder are placed in evaluation mode:

```text
Encoder → eval mode
Decoder → eval mode
```

Gradient computation is disabled during inference because the model parameters are not being updated.

This reduces unnecessary computation and memory usage.

---

# Deployment

The Flask application has been deployed using **Render**.

The deployment configuration uses:

```text
Gunicorn
```

as the production WSGI server.

The project also contains:

```text
Procfile
```

which specifies the application startup command.

The application can be accessed through the deployed Render URL when the service is running.

---

# Deployment Consideration

Neural Style Transfer is more computationally demanding than a typical Flask application because every style-transfer request requires neural network inference.

The project therefore distinguishes between:

```text
Application Layer
        ↓
Flask + Web Interface
```

and

```text
ML Inference Layer
        ↓
VGG + AdaIN + Decoder
```

The application can run locally with the full inference pipeline, while cloud deployment depends on the available CPU/GPU resources.

This is an important practical consideration when deploying deep learning applications.

---

# Challenges Faced

During development, several practical challenges were encountered.

### 1. Large model files

Deep learning models contain significantly more data than traditional machine learning models.

The project therefore requires trained model weights such as:

```text
vgg_normalised.pth
decoder_12.pth
```

---

### 2. Computational requirements

Neural Style Transfer requires running multiple neural network operations for every request.

CPU-based cloud environments can therefore be significantly slower than GPU environments.

---

### 3. Image size and inference speed

Larger input images increase computational requirements.

To make inference more manageable, the application preprocesses images to:

```text
256 × 256
```

before passing them through the model.

---

### 4. Web application + ML integration

Another challenge was integrating the deep learning pipeline into a Flask application.

The system needs to coordinate:

```text
HTTP Request
     ↓
File Upload
     ↓
Image Processing
     ↓
PyTorch Inference
     ↓
Image Saving
     ↓
HTTP Response
```

This provided practical experience in deploying machine learning models inside a web application.

---

# What I Learned

Through this project, I gained practical experience with:

* Neural Style Transfer
* Adaptive Instance Normalization
* VGG feature extraction
* Encoder-decoder architectures
* PyTorch inference
* Image preprocessing
* Model loading
* Flask backend development
* File upload handling
* HTML template rendering
* Connecting ML inference with a web application
* Deploying ML applications
* Understanding the difference between local and cloud inference
* Handling computational constraints in ML deployment

---

# Future Improvements

Possible improvements include:

* GPU-based inference for faster generation
* Higher-resolution output generation
* Better memory management
* Asynchronous inference
* Queue-based processing for multiple users
* Progress indicators during inference
* User accounts and saved results
* More advanced style controls
* Multiple style-transfer modes
* Containerized deployment
* Dedicated GPU inference backend

---

# Project Workflow

The complete system can be summarized as:

```text
                 ┌─────────────────┐
                 │      User       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  Flask Website  │
                 └────────┬────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       Content Image            Style Image
              │                       │
              ▼                       ▼
       Preprocessing             Preprocessing
              │                       │
              ▼                       ▼
       VGG Encoder               VGG Encoder
              │                       │
              ▼                       ▼
       Content Features          Style Features
              │                       │
              └───────────┬───────────┘
                          ▼
                       AdaIN
                          │
                          ▼
                  Alpha Blending
                          │
                          ▼
                       Decoder
                          │
                          ▼
                  Stylized Image
                          │
                          ▼
                  Flask Response
                          │
                          ▼
                       User
```

---

# Interview Explanation

A concise explanation of the project is:

> **“I developed a Neural Style Transfer application using PyTorch and Adaptive Instance Normalization. The system uses a pretrained VGG-19 network to extract content and style features. AdaIN aligns the channel-wise statistics of the content features with the style features, and a trained decoder reconstructs the stylized image. I integrated this inference pipeline into a Flask web application where users can upload arbitrary content and style images and control the strength of style transfer using an alpha parameter. I also deployed the web application and worked through the practical challenges of deploying computationally intensive deep learning inference.”**

---

# References

### Research Paper

Huang, X., & Belongie, S.
**Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization.**
ICCV 2017.

### Main Concepts

* Neural Style Transfer
* Adaptive Instance Normalization
* VGG feature extraction
* Encoder-decoder architecture
* Deep learning image generation

---

# Author

**Alisha Shaikh**

B.E. Electrical Engineering
Machine Learning & AI Enthusiast

GitHub:
https://github.com/AlishaShaikh20

---

## License

This project is intended for educational and portfolio purposes.
