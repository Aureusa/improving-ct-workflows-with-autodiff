def box_text(text: str, padding=1):
    """
    Wraps a string in a box.

    :param text: The text to be boxed.
    :type text: str
    :param padding: The number of spaces to pad on each side of the text. Defaults to 1.
    :type padding: int, optional
    :return: The boxed text.
    :rtype: str
    """
    lines = text.split("\n")
    width = max(len(line) for line in lines) + 2 * padding
    top = "┌" + "─" * width + "┐"
    bottom = "└" + "─" * width + "┘"

    boxed_lines = [top]
    for line in lines:
        # left-align the line and add padding
        boxed_lines.append("│" + " " * padding + line.ljust(width - 2 * padding) + " " * padding + "│")
    boxed_lines.append(bottom)
    return "\n".join(boxed_lines)
    