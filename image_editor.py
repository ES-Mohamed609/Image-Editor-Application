import io
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, Toplevel, OptionMenu, StringVar
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        self.image = None
        self.processed_image = None
        self.rotation_angle = 0  # متغير لتتبع الزاوية الحالية للدوران

        self.filter_var = StringVar(root)
        self.filter_var.set("Select Filter")
        filters = ["Grayscale",
                    "CLAHE",
                      "Histogram Equalization",
                        "BGR to RGB",
                            "Gaussian Blur",
                              "Median Filter",
                                "Laplacian",
                                "Convolution",
                                  "Correlation",
                                    "Mean Filter",
                                      "Laplacian of Gaussian",
                                        "Custom Kernel",
                                        "Invert",
                                        "Thresholding",
                                        ]
        OptionMenu(root, self.filter_var, *filters, command=self.apply_filter).pack(pady=5)

        # Buttons
        Button(root, text="Load Image", command=self.load_image).pack(pady=5)
        Button(root, text="Save Image", command=self.save_image).pack(pady=5)
        Button(root, text="Rotate Image", command=self.rotate_image).pack(pady=5)
        Button(root, text="Resize Image", command=self.resize_image).pack(pady=5)
        Button(root, text="Subtract Images", command=self.subtract_image).pack(pady=5)
        Button(root, text="Add Images", command=self.weighted_addition).pack(pady=5)
        Button(root, text="Show Color Channels", command=self.show_color_channels).pack(pady=5)

        self.image_label = Label(root)
        self.image_label.pack()

    def apply_filter(self, selection):
        if self.image is not None:
            match selection:
                case "Invert":
                    self.invert_image()
                case "Thresholding":
                    self.apply_thresholding()
                case "Grayscale":
                    self.processed_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                    self.display_image(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR))
                case "BGR to RGB":
                    self.processed_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    self.display_image(self.processed_image)
                case "Gaussian Blur":
                    self.Gaussian_filter()
                case "Median Filter":
                    self.Median_blur()
                case "Laplacian":
                    self.laplacian()
                case "Convolution":
                    kernel = np.ones((5, 5), np.float32) / 25
                    self.processed_image = cv2.filter2D(self.image, -1, kernel)
                    self.display_image(self.processed_image)
                case "Correlation":
                    kernel = np.ones((5, 5), np.float32) / 25
                    self.processed_image = cv2.filter2D(self.image, -1, kernel)  # Similar to convolution
                    self.display_image(self.processed_image)
                case "Mean Filter":
                    self.Mean_filter()
                case "Laplacian of Gaussian":
                    self.laplacianOfGaussian()
                case "Histogram Equalization":
                    self.histogram_equalization()
                case "Custom Kernel":
                    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                    self.processed_image = cv2.filter2D(self.image, -1, kernel)
                    self.display_image(self.processed_image)
                case _:
                    print(f"Unknown filter: {selection}")
    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if filepath:
            self.image = cv2.imread(filepath)
            self.processed_image = self.image.copy()
            self.rotation_angle = 0  # إعادة تعيين الزاوية عند تحميل صورة جديدة
            self.display_image(self.image)

    def save_image(self):
        if self.processed_image is not None:
            filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if filepath:
                cv2.imwrite(filepath, self.processed_image)
        else:
            self.show_message("No processed image to save.")

    def display_image(self, img):
        width, height = 300,300
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img_rgb, (width, height))
        img_pil = Image.fromarray(resized_image)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def rotate_image(self):
        if self.processed_image is not None:
            self.rotation_angle = (self.rotation_angle + 90) % 360  # زيادة الزاوية بـ 90 درجة
            center = (self.processed_image.shape[1] // 2, self.processed_image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
            height, width = self.processed_image.shape[:2]
            self.processed_image = cv2.warpAffine(self.image, rotation_matrix, (width, height))
            self.display_image(self.processed_image)
        else:
            self.show_message("No image loaded.")

    def resize_image(self):
        if self.image is not None:
            width, height = 300, 300  # New dimensions
            self.processed_image = cv2.resize(self.image, (width, height))
            self.display_image(self.processed_image)
        else:
            self.show_message("No image loaded.")

    def show_color_channels(self):
      """Show individual color channels."""
      if self.image is not None:
        b, g, r = cv2.split(self.image)
        channels = {'Blue': b, 'Green': g, 'Red': r}

        for color, channel in channels.items():
            popup = Toplevel(self.root)
            popup.title(f"{color} Channel")
            channel_image = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
            self.display_image_in_popup(channel_image, popup)
      else:
        self.show_message("No image loaded.")

    def apply_thresholding(self):
      """Apply Thresholding to the image."""
      if self.image is not None:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        threshold_value = 127  # قيمة العتبة
        _, self.processed_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        self.display_image(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR))
      else:
        self.show_message("No image loaded.")

    def show_color_channels(self):
      """Show individual color channels."""
      if self.image is not None:
        b, g, r = cv2.split(self.image)
        channels = {'Blue': b, 'Green': g, 'Red': r}

        for color, channel in channels.items():
            popup = Toplevel(self.root)
            popup.title(f"{color} Channel")
            channel_image = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
            self.display_image_in_popup(channel_image, popup)
      else:
        self.show_message("No image loaded.")

    def invert_image(self):
      """Invert image colors."""
      if self.image is not None:
        self.processed_image = cv2.bitwise_not(self.image)
        self.display_image(self.processed_image)
      else:
        self.show_message("No image loaded.")

    def histogram_equalization(self):
      if self.image is not None:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray)
        plt.figure(figsize=(12, 8))
        
        # Oiginal image
        plt.subplot(2, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Original Image")
        plt.axis("off")
        
        # Histogram Of Original Image
        plt.subplot(2, 2, 2)
        #plt.plot(original_hist, color='blue')
        plt.hist(equalized_image.ravel(), 256, [0,256])
        plt.title("Histogram of Original Image")
        
        # CLAHE Equalized
        plt.subplot(2, 2, 3)
        plt.imshow(equalized_image, cmap='gray')
        plt.title("Image after Histogram Equalization")
        plt.axis("off")
        
        # Histogram Of CLAHE Equalized Image
        plt.subplot(2, 2, 4)
        plt.hist(equalized_image.ravel(), 256, [0, 256])
        plt.title("Histogram after Equalization")
        plt.tight_layout()
        plt.show()
      else:
        self.show_message("No image loaded.")

    def apply_clahe(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.processed_image = clahe.apply(gray)
            self.display_image(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR))
        else:
            self.show_message("No image loaded.")

    def weighted_addition(self):
        if self.image is not None:
            alpha, beta = 0.5, 0.5
            second_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if second_image_path:
                second_image = cv2.imread(second_image_path)
                second_image_resized = cv2.resize(second_image, (self.image.shape[1], self.image.shape[0]))
                self.processed_image = cv2.addWeighted(self.image, alpha, second_image_resized, beta, 0)
                self.display_image(self.processed_image)
        else:
            self.show_message("No image loaded.")

    def show_message(self, message):
        popup = Toplevel(self.root)
        popup.title("Message")
        Label(popup, text=message).pack(pady=10)
        Button(popup, text="OK", command=popup.destroy).pack(pady=5)

    ## Sharpen

    def laplacian(self):
        if self.image is not None:
            laplacian = cv2.Laplacian(self.image, -1, (5,5))
            self.processed_image = laplacian
            self.display_image(self.processed_image)
        else:
            self.show_message("No image loaded.")
    
    def laplacianOfGaussian(self):
        if self.image is not None:
            gaussian = cv2.GaussianBlur(self.image, (3,3), 0)
            laplacian = cv2.Laplacian(gaussian, -1, (5,5))
            self.processed_image = laplacian
            self.display_image(self.processed_image)
        else:
            self.show_message("No image loaded.")

    def sharpenLaplacian(self):
        if self.image is not None:
            laplacian = cv2.Laplacian(self.image, -1, (5,5))
            laplacian = cv2.convertScaleAbs(laplacian)
            self.processed_image = cv2.addWeighted(self.image, 1, laplacian, -0.5, 0)
            self.display_image(self.processed_image)

    ## Smoothing
    def Mean_filter(self):
        if self.image is not None:
            kernel_size = (5,5)
            self.processed_image = cv2.blur(self.image, kernel_size)
            self.display_image(self.processed_image)
        else:
            self.show_message("No image loaded.")
    def Weighted_avr_filter(self):
        if self.image is not None:
            kernel1 = 1 / 16 * (np.array([[1, 2, 1],
                                          [2, 4, 2],
                                          [1, 2, 1]]))
            self.processed_image = cv2.filter2D(self.image, -1, kernel1)
            self.display_image(self.processed_image)
        else:
            self.show_message("No image loaded.")
    def Median_blur(self):
        if self.image is not None:
            #kernel1 = 5
            self.processed_image = cv2.medianBlur(self.image,  5)
            self.display_image(self.processed_image)
        else:
            self.show_message("No image loaded.")
    def Gaussian_filter(self):
        if self.image is not None:
            kernel1 = (5,5)
            self.processed_image = cv2.GaussianBlur(self.image, kernel1, 21)
            self.display_image(self.processed_image)
        else:
            self.show_message("No image loaded.")

    def subtract_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if filepath:
            img2 = cv2.imread(filepath)
            height, width = self.image.shape[:2]
            img2 = cv2.resize(img2, (width, height))
            self.processed_image = cv2.subtract(self.image, img2)
            self.display_image(self.processed_image)

    def display_image_in_popup(self, img, popup):
      """Display image in a popup window."""
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      resized_image = cv2.resize(img_rgb, (300, 300))
      img_pil = Image.fromarray(resized_image)
      img_tk = ImageTk.PhotoImage(img_pil)
      Label(popup, image=img_tk).pack()
      popup.image = img_tk

# Main Application
root = tk.Tk()
app = ImageEditor(root)
root.mainloop()
