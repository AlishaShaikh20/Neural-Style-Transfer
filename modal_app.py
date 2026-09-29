
import modal
import torch
import io
import base64
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------
# 1. Build the Modal container
# ---------------------------------------------------------

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi[standard]",
        "torch",
        "torchvision",
        "Pillow",
    )
    .add_local_dir(
        "utils",
        "/root/utils"
    )
    .add_local_file(
        "vgg/vgg_normalised.pth",
        "/root/vgg/vgg_normalised.pth"
    )
    .add_local_file(
        "decoder_12.pth",
        "/root/decoder_12.pth"
    )
)


# ---------------------------------------------------------
# 2. Create Modal App
# ---------------------------------------------------------

app = modal.App(
    "adain-nst-app",
    image=image
)


# ---------------------------------------------------------
# 3. Model container
# ---------------------------------------------------------

@app.cls()
class AdaINModel:

    @modal.enter()
    def load_model(self):

        print("Loading AdaIN model...")

        self.device = torch.device("cpu")

        # Load VGG encoder
        self.encoder = VGGEncoder(
            "/root/vgg/vgg_normalised.pth"
        ).to(self.device)

        # Load decoder
        self.decoder = Decoder().to(self.device)

        self.decoder.load_state_dict(
            torch.load(
                "/root/decoder_12.pth",
                map_location=self.device
            )
        )

        self.encoder.eval()
        self.decoder.eval()

        print("AdaIN model loaded successfully!")


    # -----------------------------------------------------
    # 4. AdaIN style transfer
    # -----------------------------------------------------

    def style_transfer(
        self,
        content_image,
        style_image,
        alpha=1.0
    ):

        content_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor()
        ])

        style_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor()
        ])

        content_image = (
            content_transform(content_image)
            .unsqueeze(0)
            .to(self.device)
        )

        style_image = (
            style_transform(style_image)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            content_feats = self.encoder(
                content_image,
                is_test=True
            )

            style_feats = self.encoder(
                style_image,
                is_test=True
            )

            stylized_feats = adaptive_instance_normalization(
                content_feats,
                style_feats
            )

            stylized_feats = (
                alpha * stylized_feats
                + (1 - alpha) * content_feats
            )

            stylized_image = self.decoder(
                stylized_feats
            )

        return stylized_image


    # -----------------------------------------------------
    # 5. Generate style transfer result
    # -----------------------------------------------------

    @modal.fastapi_endpoint(method="POST")
    def generate_style(self, data: dict):

        # Get base64 images
        content_b64 = data.get(
            "content_image",
            ""
        ).split(",")[-1]

        style_b64 = data.get(
            "style_image",
            ""
        ).split(",")[-1]

        # Convert base64 content image → PIL
        content_img = Image.open(
            io.BytesIO(
                base64.b64decode(content_b64)
            )
        ).convert("RGB")

        # Convert base64 style image → PIL
        style_img = Image.open(
            io.BytesIO(
                base64.b64decode(style_b64)
            )
        ).convert("RGB")

        # Get alpha
        alpha = float(
            data.get("alpha", 1.0)
        )

        print("Running AdaIN inference...")

        # Run AdaIN
        stylized_img = self.style_transfer(
            content_img,
            style_img,
            alpha
        )

        # Tensor → PIL
        stylized_img = stylized_img.cpu().clone()
        stylized_img = stylized_img.squeeze(0)
        stylized_img = stylized_img.clamp(0, 1)

        stylized_img = transforms.ToPILImage()(
            stylized_img
        )

        # PIL → Base64
        buffered = io.BytesIO()

        stylized_img.save(
            buffered,
            format="JPEG"
        )

        output_b64 = base64.b64encode(
            buffered.getvalue()
        ).decode("utf-8")

        print("AdaIN inference completed!")

        return {
            "status": "success",
            "result_image": (
                f"data:image/jpeg;base64,{output_b64}"
            )
        }


# ---------------------------------------------------------
# 6. Import your AdaIN code
# ---------------------------------------------------------

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

