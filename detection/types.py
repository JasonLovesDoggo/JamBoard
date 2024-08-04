import dataclasses


@dataclasses.dataclass
class ShapeData:
    size: int  # area
    color: tuple[int, int, int]  # RGB (or BGR)
    center: tuple[int, int]  # (cx, cy)
    name: str  # shape name (e.g., "Circle")

    def __str__(self):
        return (
            f"{self.name} at {self.center} with size {self.size} and color {self.color}"
        )


# converting function

# def to_named_color(self):
#     ...
