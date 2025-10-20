COLUMN_NAMES = [
    "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination",
    "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions",
    "Care of Magical Creatures", "Charms", "Flying"
]


def get_description():
    des = "Draw a scatter plot for the selected columns:\n"
    for c in COLUMN_NAMES:
        des += f"  {c}\n"
    return des
