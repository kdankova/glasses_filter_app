
# Face Filter App

## Overview

The Face Filter App is a Python-based desktop application that allows users to load an image and apply fun filters to it, including adding glasses and a hat. Users can also select different colors for these accessories and save the modified image.

## Features

- **Load Image:** Upload an image file (supports `.jpg`, `.jpeg`, `.png`, `.bmp` formats).
- **Add Accessories:** Add glasses and a hat to the face in the image.
- **Color Selection:** Choose from a variety of colors for the accessories (Red, Purple, Green, Yellow).
- **Save Image:** Save the modified image with the applied filters.

## Installation

To run this project, you'll need to have Python installed on your machine along with a few dependencies. You can install the necessary packages using pip.

### Dependencies

- `tkinter`
- `Pillow`
- `opencv-python`
- `numpy`
- `matplotlib`

You can install these dependencies using the following command:

```bash
pip install Pillow opencv-python numpy matplotlib
```

Note: `tkinter` is included with most Python installations. If you encounter issues, ensure that it's installed by following the appropriate steps for your operating system.

## Usage

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/yourusername/face-filter-app.git
   cd face-filter-app
   ```

2. Run the Python script to start the application.

   ```bash
   python main3.py
   ```

3. Use the GUI to:
   - Load an image.
   - Apply filters by adding glasses and a hat.
   - Select the color of the accessories.
   - Save the filtered image.

## How It Works

1. **Loading an Image:** The application uses OpenCV to load and display the image within the Tkinter window.
2. **Adding Accessories:** When the user clicks the "Add Glasses and Hat" button, the application overlays these accessories onto the face in the image.
3. **Color Selection:** The user can select the color of the accessories using a dropdown menu.
4. **Saving the Image:** The modified image can be saved to the user's file system.

## Acknowledgements

- [OpenCV](https://opencv.org/) for the powerful image processing library.
- [Pillow](https://python-pillow.org/) for handling image files.
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for the GUI framework.
