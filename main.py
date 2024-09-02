import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt


class FaceFilterApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Filter App")
        self.root.geometry("800x800")

        self.image = None
        self.filtered_image = None

        # Image upload button
        self.load_button = ttk.Button(self.root, text="Load Photo", command=self.load_image)
        self.load_button.pack(side="top", pady=10)

        # Add a hat button
        self.hat_button = ttk.Button(self.root, text="Add Glasses and Hat", command=self.add_hat)
        self.hat_button.pack(side="top", padx=10)

        # ComboBox for color selection
        self.color_combo = ttk.Combobox(self.root, values=["Red", "Purple", "Green", "Yellow"])
        self.color_combo.pack(side="top", pady=10)

        # Apply filter button
        self.apply_button = ttk.Button(self.root, text="Apply Filter", command=self.apply_filter)
        self.apply_button.pack(side="top", pady=10)

        self.save_button = ttk.Button(self.root, text="Save Photo", command=self.save_image)
        self.save_button.pack(pady=10)

        # Image display area
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(padx="200", fill="both", expand=True)

    def load_image(self):
        path = tk.filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.image = cv2.imread(path)
            self.display_image(self.image)

    def display_image(self, image):
        # Resize the image to the relevant dimensions
        resized_image = cv2.resize(image, (400, 500))
        pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

        # Displaying the image in a Tkinter Label
        image_tk = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk

    def resize_image(self, image, width=None, height=None):
        # Perform image resizing operations
        if len(image.shape) > 2:
            height, curr_width, _ = image.shape
        else:
            height, curr_width = image.shape

        ratio = width / curr_width
        new_height = int(height * ratio)
        resized_image = cv2.resize(image, (width, new_height))

        return resized_image

    def create_hair_mask(self):
        height, width = self.image.shape[:2]
        # Create mask image
        hair_mask = np.zeros((height, width), dtype=np.uint8)

        # Determine the color range to create the hair mask
        lower_hair_color = np.array([0, 0, 0])  # Minimum color value
        upper_hair_color = np.array([50, 50, 100])  # Maximum color value

        # Mark pixels that match the color range in the mask image
        hair_mask = cv2.inRange(self.image, lower_hair_color, upper_hair_color)

        cv2.imwrite('mask.jpg', hair_mask)
        print("Mask saved as mask.jpg")
        return hair_mask

    def change_hair_color(self, image, hair_mask, new_hair_color, opacity):
        # Convert to HSV format for color conversion
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Applying the hair mask
        hair_mask = cv2.cvtColor(hair_mask, cv2.COLOR_BGR2GRAY)
        _, hair_mask = cv2.threshold(hair_mask, 10, 255, cv2.THRESH_BINARY)

        # Selecting hair areas
        hair_pixels = np.where(hair_mask[:, :] > 0)
        # Changing hair color
        if new_hair_color == 'Red':
            hsv_image[hair_pixels[0], hair_pixels[1], 0] = 0  # Hue value (hue) is made 0 (red)
            hsv_image[hair_pixels[0], hair_pixels[
                1], 1] = 255  # Saturation value (saturation) is set to 255 to make the hair look more vibrant
            hsv_image[hair_pixels[0], hair_pixels[
                1], 2] = opacity  # Value (brightness) is set to 255 to make the hair look brighter
        elif new_hair_color == 'Purple':
            hsv_image[hair_pixels[0], hair_pixels[1], 0] = 150  # Hue value (hue) is made 150 (purple)
            hsv_image[hair_pixels[0], hair_pixels[
                1], 1] = 255  # Saturation value (saturation) is set to 255 to make the hair look more vibrant
            hsv_image[hair_pixels[0], hair_pixels[
                1], 2] = opacity  # Value (brightness) is set to 255 to make the hair look brighter
        elif new_hair_color == 'Green':
            hsv_image[hair_pixels[0], hair_pixels[1], 0] = 60  # Hue value (hue) is made 60 (green)
            hsv_image[hair_pixels[0], hair_pixels[
                1], 1] = 255  # Saturation (saturation) is set to 255 to make the hair look more vibrant
            hsv_image[hair_pixels[0], hair_pixels[
                1], 2] = opacity  # Value  (brightness) is set to 255 to make the hair look brighter
        elif new_hair_color == 'Yellow':
            hsv_image[hair_pixels[0], hair_pixels[1], 0] = 30  # Hue value (hue) is made 30 (yellow)
            hsv_image[hair_pixels[0], hair_pixels[
                1], 1] = 255  # Saturation value (saturation) is set to 255 to make the hair look more vibrant
            hsv_image[hair_pixels[0], hair_pixels[
                1], 2] = opacity  # Value (brightness) is set to 255 to make the hair look brighter

        # Convert color conversion back to BGR format
        new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # Resize the hair mask to the original image size
        hair_mask_resized = cv2.resize(hair_mask, (new_image.shape[1], new_image.shape[0]))
        hair_mask_resized = cv2.cvtColor(hair_mask_resized, cv2.COLOR_GRAY2BGR)

        # Superimpose the hair on the original image with pixels in red color
        result = np.where(hair_mask_resized > 0, new_image, image)

        return result

    def add_hat(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        glass_img = cv2.imread('glasses.png', -1)
        hat_img = cv2.imread('hat.png', -1)
        if self.image is not None and self.image.any():
            # Adding a hat
            # Complete the code
            # Convert to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # Add glasses and hats for each face
            for (x, y, w, h) in faces:
                # Resize and position the glasses
                resized_glasses = self.resize_image(glass_img, w)
                glass_y = int(y + h / 2) - int(resized_glasses.shape[0] / 2)
                glass_x = x
                glass_h, glass_w = resized_glasses.shape[:2]

                # Add the glasses to the image (taking into account the alpha channel)
                for c in range(3):
                    self.image[glass_y:glass_y + glass_h, glass_x:glass_x + glass_w, c] = (
                            resized_glasses[:, :, c] * (resized_glasses[:, :, 3] / 255.0) +
                            self.image[glass_y:glass_y + glass_h, glass_x:glass_x + glass_w, c] * (
                                        1.0 - resized_glasses[:, :, 3] / 255.0)
                    )

                # Resize and reposition the hat
                resized_hat = self.resize_image(hat_img, width=w)
                hat_y = y - int(resized_hat.shape[0] * 0.8)
                hat_x = x
                hat_h, hat_w = resized_hat.shape[:2]

                # Add the hat image (taking into account the alpha channel)
                for c in range(3):
                    self.image[hat_y:hat_y + hat_h, hat_x:hat_x + hat_w, c] = (
                            resized_hat[:, :, c] * (resized_hat[:, :, 3] / 255.0) +
                            self.image[hat_y:hat_y + hat_h, hat_x:hat_x + hat_w, c] * (
                                        1.0 - resized_hat[:, :, 3] / 255.0)
                    )
            # Update photo with added hat
            self.filtered_image = self.image.copy()
            self.display_image(self.filtered_image)

    def apply_filter(self):
        selected_color = self.color_combo.get()
        hair_mask = self.create_hair_mask()
        hair_mask = cv2.imread("mask.jpg")
        opacity = 100
        new_image = self.change_hair_color(self.image, hair_mask, selected_color, opacity)
        self.filtered_image = new_image.copy()
        self.display_image(self.filtered_image)

    def save_image(self):
        if self.filtered_image is not None and self.filtered_image.any():
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG Image", "*.jpg")])
            if save_path:
                filtered_image_rgb = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2RGB)  # Fix color channels
                image = Image.fromarray(filtered_image_rgb)
                image.save(save_path)
                print("Photo saved successfully.")
        else:
            print("Upload a photo first and apply the filter.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceFilterApp()
    app.run()
