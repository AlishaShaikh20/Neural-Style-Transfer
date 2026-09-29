
import modal


# ---------------------------------------------------------
# 1. Build the container with only the files Flask needs
# ---------------------------------------------------------

image = (
    modal.Image.debian_slim()
    .pip_install(
        "Flask",
        "Flask-WTF",
        "Flask-SQLAlchemy",
        "Flask-Bootstrap",
        "Pillow",
        "torch",
        "torchvision",
    )
    .add_local_file(
        "app.py",
        "/root/project/app.py"
    )
    .add_local_dir(
        "templates",
        "/root/project/templates"
    )
    .add_local_dir(
        "static",
        "/root/project/static"
    )
    .add_local_dir(
    "examples",
    "/root/project/examples"
)
    .add_local_dir(
        "utils",
        "/root/project/utils"
    )
    .add_local_file(
        "vgg/vgg_normalised.pth",
        "/root/project/vgg/vgg_normalised.pth"
    )
    .add_local_file(
        "decoder_12.pth",
        "/root/project/decoder_12.pth"
    )
)


# ---------------------------------------------------------
# 2. Create Modal app
# ---------------------------------------------------------

app = modal.App(
    "adain-nst-website",
    image=image,
)


# ---------------------------------------------------------
# 3. Deploy the existing Flask application
# ---------------------------------------------------------

@app.function()
@modal.wsgi_app()
def flask_app():

    import sys

    sys.path.insert(
        0,
        "/root/project"
    )

    from app import app

    return app
