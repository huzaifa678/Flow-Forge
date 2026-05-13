import base64

def decode_base64_image(image_b64: str) -> bytes:
    """
    Decode base64 image string into raw bytes.
    """
    if not image_b64:
        return b""

    # remove data URI prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    return base64.b64decode(image_b64)