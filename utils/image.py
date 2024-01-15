def get_alpha_from_opacity(opacity):
    # Opacity should be 0 -> 0, 100 -> 255
    return int(opacity * 255 / 100)


def get_rgba(color, opacity):
    # if color is hex, convert to rgb
    if not isinstance(color, tuple) and color.startswith("#"):
        color = color.lstrip("#")
        color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    return color[0], color[1], color[2], get_alpha_from_opacity(opacity)


def get_bgra(color, opacity):
    # if color is hex, convert to rgb
    if not isinstance(color, tuple) and color.startswith("#"):
        color = color.lstrip("#")
        color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    return color[2], color[1], color[0], get_alpha_from_opacity(opacity)
