from PIL import Image
from typing import AnyStr
from potrace import Bitmap, Path


class PngToSVG(object):
    def __init__(self, png_path: AnyStr, conversion: AnyStr):
        image = Image.open(f"{png_path}.png").convert(conversion)
        bitmap = Bitmap(image)

        # Trace to path
        path_plist: Path = bitmap.trace()

        # Export to SVG
        with open(f"{png_path}.svg", "w") as f:
            PngToSVG.to_svg(path_plist, image, f)

    @staticmethod
    def to_svg(path: Path, image: Image, fp) -> None:
        fp.write(
            f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}">''')
        parts = []
        for curve in path:
            fs = curve.start_point
            parts.append(f"M{fs.x},{fs.y}")
            for segment in curve.segments:
                if segment.is_corner:
                    a = segment.c
                    b = segment.end_point
                    parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                else:
                    a = segment.c1
                    b = segment.c2
                    c = segment.end_point
                    parts.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")
            parts.append("z")
        fp.write(f'<path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(parts)}"/>')
        fp.write("</svg>")


if __name__ == '__main__':
    path = '../../images/test1'
    conversion = "1"
    PngToSVG(path, conversion)
