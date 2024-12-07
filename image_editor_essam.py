import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, OptionMenu, StringVar
from PIL import Image, ImageTk
import numpy as np


class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")
        self.image = None
        self.processed_image = None

        # Buttons
        Button(root, text="Load Image", command=self.load_image).pack(pady=5)
        Button(root, text="Save Image", command=self.save_image).pack(pady=5)

        # Dropdown for filters
        self.filter_var = StringVar(root)
        self.filter_var.set("Select Filter")
        filters = ["Grayscale", "BGR to RGB", "Binary", "Gaussian Blur", "Median Filter", "Laplacian",
                   "Convolution", "Correlation", "Mean Filter", "Laplacian of Gaussian", "Custom Kernel"]
        OptionMenu(root, self.filter_var, *filters, command=self.apply_filter).pack(pady=5)

        Button(root, text="Resize Image", command=self.resize_image).pack(pady=5)
        Button(root, text="Rotate Image", command=self.rotate_image).pack(pady=5)
        Button(root, text="Split & Merge Channels", command=self.split_merge_channels).pack(pady=5)
        Button(root, text="Apply Histogram Equalization", command=self.histogram_equalization).pack(pady=5)
        Button(root, text="Add Image", command=self.add_image).pack(pady=5)
        Button(root, text="Subtract Image", command=self.subtract_image).pack(pady=5)

        self.image_label = Label(root)
        self.image_label.pack()

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if filepath:
            self.image = cv2.imread(filepath)
            self.processed_image = self.image.copy()
            self.display_image(self.image)

    def save_image(self):
        if self.processed_image is not None:
            filepath = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if filepath:
                cv2.imwrite(filepath, self.processed_image)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    

    def resize_image(self):
        if self.image is not None:
            self.processed_image = cv2.resize(self.image, (300, 300))
            self.display_image(self.processed_image)

    def rotate_image(self):
        if self.processed_image is not None:
            center = (self.image.shape[1] // 2, self.image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
            self.processed_image = cv2.warpAffine(self.image, rotation_matrix,
                                                  (self.image.shape[1], self.image.shape[0]))
            self.display_image(self.processed_image)

    def split_merge_channels(self):
        if self.image is not None:
            b, g, r = cv2.split(self.image)
            self.display_image(cv2.merge([r, g, b]))

    def histogram_equalization(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.equalizeHist(gray)
            self.display_image(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR))

    def add_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if filepath:
            img2 = cv2.imread(filepath)
            if self.image.shape == img2.shape:
                self.processed_image = cv2.add(self.image, img2)
                self.display_image(self.processed_image)

    def subtract_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if filepath:
            img2 = cv2.imread(filepath)
            height, width = self.image.shape[:2]
            img2 = cv2.resize(img2, (width, height))
            self.processed_image = cv2.subtract(self.image, img2)
            self.display_image(self.processed_image)


# Main Application
root = tk.Tk()
app = ImageEditor(root)
root.mainloop()
