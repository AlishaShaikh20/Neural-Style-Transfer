# Neural Style Transfer

A deep learning-based **Neural Style Transfer (NST)** application that combines the content of one image with the artistic style of another image.

The project uses **Adaptive Instance Normalization (AdaIN)** to perform arbitrary style transfer and provides a **Flask web interface** where users can upload content and style images and generate a stylized result.

The application has also been deployed using **Modal**, allowing the Flask website and neural style-transfer inference pipeline to run in the cloud.

---

## Project Overview

Neural Style Transfer is a computer vision technique that transfers the visual characteristics of one image onto another image while preserving the main content and structure of the original image.

```text
Content Image (photograph of a person)
        +
Style Image (artistic painting)
        ↓
Stylized Image
```

This project implements an AdaIN-based approach inspired by:

> Huang, X., & Belongie, S.
> **Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization**
> ICCV 2017

Unlike traditional optimization-based neural style transfer, where an image may need to be optimized separately for every content-style pair, AdaIN enables arbitrary content and style images to be processed using a trained encoder-decoder architecture.

---

## Demo

The Flask-based web application lets users:

* Upload a content image
* Upload a style image
* Select the strength of style transfer using an alpha value
* Generate a stylized image
* View the content, style, and generated images
* Explore predefined examples
* View project information and FAQs

### Live Demo

**NeuralArt / StyleForge AI**

https://alishashaikh20--adain-nst-website-flask-app.modal.run

The deployed application runs the Flask web interface together with the required neural network model files.

---

## Key Features

* Arbitrary content and style image selection
* Adaptive Instance Normalization (AdaIN)
* VGG-based feature extraction
* Trained decoder for image reconstruction
* Adjustable style-transfer strength
* Flask web interface
* Image upload and validation
* Example gallery
* Cloud deployment using Modal
* CPU-based inference support
* Separate Modal inference endpoint for testing the AdaIN model

---

## How Neural Style Transfer Works

```text
Content Image ──► Resize & Preprocess ──► VGG Encoder ──► Content Features ──┐
                                                                              ├──► AdaIN ──► Alpha Interpolation ──► Decoder ──► Stylized Image
Style Image   ──► Resize & Preprocess ──► VGG Encoder ──► Style Features   ──┘
```

The model does not directly combine the pixels of the two images. Instead, it performs the style transfer in **feature space**.

### Adaptive Instance Normalization

AdaIN modifies the content feature statistics so that their channel-wise mean and variance match those of the style features:

```text
AdaIN(x, y) = σ(y) * ((x - μ(x)) / σ(x)) + μ(y)
```

Where:

* `x` = content feature map
* `y` = style feature map
* `μ(x)`, `σ(x)` = channel-wise mean and standard deviation of content features
* `μ(y)`, `σ(y)` = channel-wise mean and standard deviation of style features

The result is a transformed content feature representation whose statistics resemble the selected style. This is the key mechanism that allows the application to accept arbitrary style images instead of requiring a separate model for every artistic style.

### Why VGG?

The project uses a pretrained **VGG-19-based encoder** for feature extraction. VGG is not being used here primarily as an image classifier. Instead, intermediate convolutional features represent visual information such as edges, shapes, textures, patterns, object structures, and higher-level visual characteristics.

The same encoder is used for both the content and style images, and its parameters remain fixed during inference.

### Encoder

The custom `VGGEncoder` lives in `utils/models.py`. It:

1. Loads the pretrained VGG weights.
2. Processes input images.
3. Extracts intermediate feature representations.
4. Returns the feature representation required by AdaIN.

The encoder is loaded once when the Flask application starts and used with `encoder.eval()`.

### Decoder

After AdaIN produces the stylized feature representation, the decoder reconstructs the final image. Its learned weights are stored in `decoder_12.pth`.

### Alpha Parameter

The `alpha` parameter controls the strength of the style transformation:

```text
stylized_features = alpha × stylized_features + (1 - alpha) × content_features
```

* `alpha = 1.0` — the output uses the fully stylized feature representation
* `alpha = 0.0` — the original content features are retained
* `alpha = 0.5` — an interpolation between the original content and stylized features

---

## Image Preprocessing

Before inference, both images are:

1. Loaded using Pillow.
2. Converted to RGB.
3. Resized to a maximum size of `512` according to the torchvision `Resize(512)` transformation.
4. Converted into PyTorch tensors.
5. Given a batch dimension.
6. Moved to the available device.

The same preprocessing is applied to the content and style images.

---

## Inference Pipeline

1. **Receive images** — the user uploads a content and a style image through the Flask interface.
2. **Validate files** — only `png`, `jpg`, `jpeg` are allowed; filenames pass through `secure_filename()`.
3. **Save uploads** — files are stored in `static/uploads/`.
4. **Load images** — `Image.open(path).convert('RGB')`.
5. **Preprocess** — resize and convert to tensors.
6. **Extract features** — the VGG encoder produces `content_features` and `style_features`.
7. **Apply AdaIN** — style statistics are transferred to the content features.
8. **Apply alpha** — stylized features are interpolated with the content features.
9. **Decode** — the decoder reconstructs the stylized image.
10. **Save result** — tensor → CPU → clamp values → remove batch dimension → PIL Image → save.
11. **Display result** — Flask passes the generated filename to the HTML template.

---

## Web Application Architecture

The project uses **Flask** as the backend. It handles HTTP requests, file uploads and validation, image preprocessing, model inference, output generation, serving uploaded and example images, and rendering HTML templates.

---

## Project Structure

```text
Neural-Style-Transfer/
│
├── app.py
├── modal_app.py
├── deploy_flask.py
│
├── decoder_12.pth
├── requirements.txt
│
├── templates/
│   └── index.html
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

## Important Files

### `app.py`
The main Flask application. Contains Flask configuration, the upload form, file validation, model loading, image preprocessing, the style-transfer function, image saving, and routes for the page, uploaded images, and example images.

### `utils/models.py`
The model architecture: the VGG encoder and the decoder.

### `utils/utils.py`
Utility functions required by the model, most importantly `adaptive_instance_normalization()`, which performs the AdaIN operation.

### `decoder_12.pth`
The trained decoder weights, loaded when the Flask application starts.

### `vgg/vgg_normalised.pth`
The pretrained VGG weights used by the encoder.

### `modal_app.py`
The separate Modal deployment for the AdaIN inference API. It builds a Modal image, installs PyTorch and dependencies, copies model and utility files, loads the encoder and decoder, runs CPU inference, and exposes a FastAPI-compatible endpoint that accepts base64-encoded content and style images and returns the generated image as base64. It was tested successfully with a content image and La Muse.

### `deploy_flask.py`
The Modal deployment wrapper for the complete Flask website. Instead of uploading the entire project directory, it explicitly includes `app.py`, `templates/`, `static/`, `examples/`, `utils/`, `vgg/vgg_normalised.pth`, and `decoder_12.pth`, keeping the deployment package focused on what the application actually needs.

---

## Technologies Used

* **Programming:** Python
* **Deep Learning:** PyTorch, Torchvision, VGG-19, Adaptive Instance Normalization, encoder-decoder architecture
* **Computer Vision:** Pillow, image preprocessing, feature extraction
* **Web Development:** Flask, Flask-WTF, Flask-Bootstrap, HTML, CSS
* **Deployment:** Modal, Flask WSGI application, FastAPI-compatible inference endpoint

---

## Installation

```bash
git clone https://github.com/AlishaShaikh20/Neural-Style-Transfer.git
cd Neural-Style-Transfer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running Locally

```bash
python app.py
```

The application will run at `http://127.0.0.1:5000`. Then:

1. Upload a content image.
2. Upload a style image.
3. Select an alpha value.
4. Click **Transfer Style**.
5. Wait for inference to complete.
6. View the generated result.

---

## Deployment with Modal

The project uses Modal to deploy the Flask application and neural network inference environment. The deployment contains the required model files and project directories inside the Modal container, and the Flask application is exposed using a WSGI wrapper.

Deployed at: `https://alishashaikh20--adain-nst-website-flask-app.modal.run`

```text
User → Public Modal URL → Flask Application (templates, static files, examples, uploads)
     → PyTorch Model (VGG Encoder → AdaIN → Decoder) → Generated Image
```

### CPU Inference

The deployed application was tested using CPU inference. The model is loaded once when the container starts and placed in evaluation mode. Inference runs under `torch.no_grad()` because model parameters are not updated.

CPU inference is functional but slower than GPU inference, so the application is primarily intended as a demonstration and portfolio project rather than a high-throughput production service.

---

## Challenges Faced and Solutions

### 1. Large Model Files
**Problem:** `vgg_normalised.pth` and `decoder_12.pth` are much larger than ordinary source files.
**Solution:** The deployment explicitly includes the required model files rather than uploading unrelated project files.

### 2. Modal Deployment Initially Included Too Many Files
**Problem:** The first approach uploaded the entire project directory, resulting in a very large number of files.
**Solution:** Changed to explicitly include only `app.py`, `templates/`, `static/`, `examples/`, `utils/`, `vgg/vgg_normalised.pth`, and `decoder_12.pth`.

### 3. Flask-Bootstrap Dependency Error
**Problem:** The deployed app initially returned `Method Not Allowed` because the deployment environment did not contain Flask-Bootstrap.
**Solution:** Added `Flask-Bootstrap` to the Modal image dependencies. After redeployment the app loaded correctly.

### 4. Gallery Examples Were Missing After Deployment
**Problem:** The website loaded, but the example images were unavailable because `examples/` had not been included.
**Solution:** Updated the Modal Flask deployment to include:

```python
.add_local_dir("examples", "/root/project/examples")
```

### 5. Uploaded Image Path Error
**Problem:** The deployed app produced `[Errno 2] No such file or directory: 'static/uploads/Jungkook.jpg'`. The original config used a relative path that depended on the current working directory:

```python
app.config['UPLOAD_FOLDER'] = 'static/uploads'
```

**Solution:** Use the absolute project directory, and create the folder at startup:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
```

### 6. Incorrect or Repeated Previous Results During Testing
**Problem:** The result filename was built as `'stylized_' + content_filename`, so using the same content image with a different style reused the same output filename. During browser testing, an older result could appear to be displayed.
**Current status:** The inference pipeline was verified to work correctly with different styles, including La Muse and Mondrian. A future improvement is to generate a unique filename per request (e.g. a UUID).

### 7. Style Transfer Quality Depends on the Input Image
**Observation:** Portrait-oriented content images with a clear subject often produced more coherent results than full-body subjects, large backgrounds, or complex compositions.
**Explanation:** AdaIN transfers style statistics through deep feature representations, so output quality depends on the composition of the inputs. This is treated as a model limitation rather than a deployment failure.

---

## Testing

The deployed application was tested with combinations such as:

```text
Jungkook + La Muse
Jimin + La Muse
Jungkook + The Scream
Jimin + Mondrian
```

Testing confirmed that uploaded content and style images were processed, different styles produced different outputs (La Muse and Mondrian results matched their styles), different content images worked, and the upload-path error was resolved.

---

## Limitations

* CPU inference can be slow.
* Higher-resolution inputs increase computational requirements.
* Output quality varies depending on the content and style images.
* Complex/full-body images may be less coherent than simple portraits.
* The result filename is based on the content filename and can be reused.
* The deployed environment uses temporary container storage for uploads/results.
* Designed primarily as a portfolio/demo application, not a large-scale multi-user service.

## Future Improvements

* Unique filenames for every generated result
* Better browser cache handling
* GPU-based inference and higher-resolution output
* Faster inference
* Asynchronous/background and queue-based processing
* Progress indicators
* Persistent result storage
* User accounts and saved generation history
* Multiple style-transfer modes and more advanced style controls
* Containerized production architecture with a dedicated GPU inference backend

---

## What I Learned

* Neural Style Transfer, AdaIN, VGG feature extraction, and encoder-decoder architectures
* PyTorch inference: image preprocessing, model loading, `torch.no_grad()`, evaluation mode
* Flask backend development, Flask-WTF forms, file uploads, secure filename handling, templates, static files, and routing
* Cloud deployment with Modal, WSGI deployment, and FastAPI-compatible endpoints
* Debugging cloud path issues, managing model files in deployment, and understanding browser caching and generated-result filenames

---

## References

Huang, X., & Belongie, S. **Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization.** International Conference on Computer Vision (ICCV), 2017.

---

## Author

**Alisha Shaikh**
B.E. Electrical Engineering
Machine Learning & AI Enthusiast

GitHub: https://github.com/AlishaShaikh20

---

## License

This project is intended for educational and portfolio purposes.
