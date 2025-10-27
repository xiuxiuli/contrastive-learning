import os

def detect_environment():
    if "COLAB_GPU" in os.environ or "COLAB_REALEASE_TAG" in os.environ:
        return "colab"
    return "local"
