
def get_rgba(color, opacity):
    # Opacity should be 0 -> 0, 100 -> 255
    alpha = int(opacity * 255 / 100)

    # if color is hex, convert to rgb
    if color.startswith("#"):
        color = color.lstrip("#")
        color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    return color[0], color[1], color[2], alpha