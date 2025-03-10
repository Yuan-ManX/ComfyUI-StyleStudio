from .StyleStudio_nodes import StyleStudio

NODE_CLASS_MAPPINGS = {
    "StyleStudio Image Stylization": StyleStudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleStudio": "StyleStudio Image Stylization",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
