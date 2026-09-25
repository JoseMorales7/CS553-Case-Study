from PIL import Image

def preview_upload(file_path: str | None):
    if not file_path:
        return None, None, "Upload an image to begin."

    try:
        with Image.open(file_path) as image:
            preview = image.convert("RGB")
        return preview, file_path, "Image ready for critique."
    except Exception as error:
        return None, None, f"Could not read that image: {error}"