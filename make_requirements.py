requirements = [
    "numpy",
    "opencv-python",
    "Pillow",
    "matplotlib",
    "torch",
    "torchvision",
    "ultralytics",
    "transformers",
]

with open("requirements.txt", "w", encoding="utf-8") as file:
    for package in requirements:
        file.write(package + "\n")

print("Utworzono plik requirements.txt")