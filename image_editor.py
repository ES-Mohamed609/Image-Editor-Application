import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, Toplevel
from PIL import Image, ImageTk

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        self.image = None
        self.processed_image = None
        self.rotation_angle = 0  # متغير لتتبع الزاوية الحالية للدوران

        # Buttons
        Button(root, text="Load Image", command=self.load_image).pack(pady=5)
        Button(root, text="Save Image", command=self.save_image).pack(pady=5)
        Button(root, text="Rotate Image", command=self.rotate_image).pack(pady=5)
        Button(root, text="Resize Image", command=self.resize_image).pack(pady=5)
        Button(root, text="Histogram Equalization", command=self.histogram_equalization).pack(pady=5)
        Button(root, text="CLAHE", command=self.apply_clahe).pack(pady=5)
        Button(root, text="Weighted Addition", command=self.weighted_addition).pack(pady=5)
        Button(root, text="Laplacian", command=self.laplacian).pack(pady=5)
        Button(root, text="Sharpen Laplacian", command=self.sharpenLaplacian).pack(pady=5)
        Button(root, text="Laplacian of Gaussian", command=self.laplacianOfGaussian).pack(pady=5)
        Button(root, text="Mean filter ", command=self.Mean_filter).pack(pady=5)
        Button(root, text="Weighted Averaging filter  ", command=self.Weighted_avr_filter).pack(pady=5)
        Button(root, text=" Median filter ", command=self.Median_blur).pack(pady=5)
        Button(root, text=" Gaussian filter ", command=self.Gaussian_filter).pack(pady=5)

        self.image_label = Label(root)
        self.image_label.pack()

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

    def histogram_equalization(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.equalizeHist(gray)
            self.display_image(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR))
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


# Main Application
root = tk.Tk()
app = ImageEditor(root)
root.mainloop()
