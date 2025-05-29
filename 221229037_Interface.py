import math
import os

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QFileDialog, QLabel, QSlider, QComboBox, QLineEdit, QPushButton, QWidget, QVBoxLayout, \
    QHBoxLayout
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fontTools.merge import layout


class Odev1Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ödev1 - Temel İşlevlik")
        self.resize(600, 500)
        self.setStyleSheet("background-color: white;")

        self.select_image_button = QtWidgets.QPushButton("Resim Seç", self)
        self.select_image_button.setGeometry(10, 10, 120, 40)
        self.select_image_button.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.select_image_button.clicked.connect(self.select_image)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 70, 500, 300)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        self.contrast_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.contrast_slider.setGeometry(50, 380, 200, 20)
        self.contrast_slider.setMinimum(1)
        self.contrast_slider.setMaximum(300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_image)

        self.contrast_label = QLabel("Kontrast: 1.0", self)
        self.contrast_label.setGeometry(260, 380, 120, 20)

        self.brightness_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.brightness_slider.setGeometry(50, 420, 200, 20)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_image)

        self.brightness_label = QLabel("Parlaklık: 0", self)
        self.brightness_label.setGeometry(260, 420, 120, 20)

        self.selected_image = None
        self.original_image = None

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)")
        if file_name:
            self.selected_image = file_name
            self.original_image = cv2.imread(file_name)
            self.update_image()

    def update_image(self):
        if self.selected_image is None:
            return

        contrast = self.contrast_slider.value() / 100
        brightness = self.brightness_slider.value()

        self.contrast_label.setText(f"Kontrast: {contrast:.1f}")
        self.brightness_label.setText(f"Parlaklık: {brightness}")

        adjusted_image = cv2.convertScaleAbs(self.original_image, alpha=contrast, beta=brightness)

        height, width, channel = adjusted_image.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(adjusted_image.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(500, 300, QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)


class HistogramWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram Window")
        self.resize(800, 600)
        self.setStyleSheet("background-color: white;")

        self.select_image_button = QtWidgets.QPushButton('Resim Seç', self)
        self.select_image_button.setGeometry(QtCore.QRect(10, 10, 120, 40))
        self.select_image_button.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.select_image_button.clicked.connect(self.select_image)

        self.show_histogram_button = QtWidgets.QPushButton('Histogram Göster', self)
        self.show_histogram_button.setGeometry(QtCore.QRect(10, 60, 120, 40))
        self.show_histogram_button.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.show_histogram_button.clicked.connect(self.show_histogram)

        self.selected_image = None
        self.threshold_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.threshold_slider.setGeometry(10, 110, 200, 20)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(128)

        self.threshold_label = QLabel("Eşik Değeri:128", self)
        self.threshold_label.setGeometry(220, 110, 150, 20)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)

        self.apply_threshold_button = QtWidgets.QPushButton('Eşikle', self)
        self.apply_threshold_button.setGeometry(QtCore.QRect(10, 140, 120, 40))
        self.apply_threshold_button.setStyleSheet("background-color: rgb(255, 85, 0);")
        self.apply_threshold_button.clicked.connect(self.apply_threshold)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)")
        if file_name:
            self.selected_image = file_name

    def show_histogram(self):
        if self.selected_image:
            image = cv2.imread(self.selected_image, cv2.IMREAD_GRAYSCALE)
            plt.hist(image.ravel(), bins=256, color='gray', alpha=0.7)
            plt.title('Resim Histogramı')
            plt.xlabel('Pixel Değeri')
            plt.ylabel('Frekans')
            plt.show()

    def update_threshold_label(self):
        self.threshold_label.setText(f"Eşik Değeri: {self.threshold_slider.value()}")

    def apply_threshold(self):
        if self.selected_image:
            image = cv2.imread(self.selected_image, cv2.IMREAD_GRAYSCALE)
            threshold_value = self.threshold_slider.value()
            _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
            cv2.imshow("Eşiklenmiş Resim", thresholded_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



class Odev2Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ödev2 - Görüntü Yakınlaştırma/Uzaklaştırma/Döndürme")
        self.resize(700, 500)
        self.setStyleSheet("background-color: white;")

        self.image_label = QLabel(self)
        self.image_label.setGeometry(100, 70, 500, 300)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        self.select_button = QtWidgets.QPushButton("Resim Ekle", self)
        self.select_button.setGeometry(30, 10, 100, 40)
        self.select_button.clicked.connect(self.load_image)

        self.resize_in_button = QtWidgets.QPushButton("Büyüt", self)
        self.resize_in_button.setGeometry(140, 10, 100, 40)
        self.resize_in_button.clicked.connect(self.resize_in)

        self.resize_out_button = QtWidgets.QPushButton("Küçült", self)
        self.resize_out_button.setGeometry(250, 10, 100, 40)
        self.resize_out_button.clicked.connect(self.resize_out)

        self.zoom_in_button = QtWidgets.QPushButton("Yakınlaştır", self)
        self.zoom_in_button.setGeometry(360, 10, 100, 40)
        self.zoom_in_button.clicked.connect(self.zoom_in)

        self.zoom_out_button = QtWidgets.QPushButton("Uzaklaştır", self)
        self.zoom_out_button.setGeometry(470, 10, 100, 40)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        self.angle_input = QLineEdit(self)
        self.angle_input.setGeometry(580, 10, 50, 40)
        self.angle_input.setPlaceholderText("Açı")

        self.rotate_button = QtWidgets.QPushButton("Döndür", self)
        self.rotate_button.setGeometry(640, 10, 60, 40)
        self.rotate_button.clicked.connect(self.rotate_image)

        self.interpolation_combo = QComboBox(self)
        self.interpolation_combo.setGeometry(50, 400, 150, 30)
        self.interpolation_combo.addItems(["Nearest", "Bilinear", "Average"])

        self.original_image = None
        self.displayed_image = None
        self.zoom_factor = 1.0

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            self.original_image = self.read_image(file_name)
            self.zoom_factor = 1.0
            self.displayed_image = self.original_image.copy()
            self.display_image(self.displayed_image)

    def read_image(self, file_path):

        return cv2.imread(file_path)

    def resize_in(self):
        if self.displayed_image is not None:
            self.displayed_image = self.resize_image(self.displayed_image, 1.2)
            self.display_image(self.displayed_image)

    def resize_out(self):
        if self.displayed_image is not None:
            self.displayed_image = self.resize_image(self.displayed_image, 0.8)
            self.display_image(self.displayed_image)

    def zoom_in(self):
        if self.original_image is not None:
            self.zoom_factor *= 1.2
            self.apply_zoom()

    def zoom_out(self):
        if self.original_image is not None:
            self.zoom_factor /= 1.2
            self.apply_zoom()

    def apply_zoom(self):
        h, w, _ = self.original_image.shape
        zoom_w = int(w / self.zoom_factor)
        zoom_h = int(h / self.zoom_factor)

        if zoom_w < 2 or zoom_h < 2:
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Daha fazla yakınlaştırılamaz.")
            return

        x1 = (w - zoom_w) // 2
        y1 = (h - zoom_h) // 2
        cropped = self.original_image[y1:y1 + zoom_h, x1:x1 + zoom_w]
        self.displayed_image = self.resize_image(cropped, self.zoom_factor)
        self.display_image(self.displayed_image)

    def resize_image(self, img, factor):
        method = self.interpolation_combo.currentText()
        if method == "Nearest":
            return self.resize_nearest(img, factor)
        elif method == "Bilinear":
            return self.resize_bilinear(img, factor)
        elif method == "Average":
            return self.resize_average(img, factor)

    def resize_nearest(self, img, factor):
        h, w, c = img.shape
        new_h = int(h * factor)
        new_w = int(w * factor)
        resized = np.zeros((new_h, new_w, c), dtype=np.uint8)
        for y in range(new_h):
            for x in range(new_w):
                src_x = int(x / factor)
                src_y = int(y / factor)
                resized[y, x] = img[min(src_y, h-1), min(src_x, w-1)]
        return resized

    def resize_bilinear(self, img, factor):
        h, w, c = img.shape
        new_h = int(h * factor)
        new_w = int(w * factor)
        resized = np.zeros((new_h, new_w, c), dtype=np.uint8)
        for y in range(new_h):
            for x in range(new_w):
                gx = x / factor
                gy = y / factor
                x0 = int(gx)
                x1 = min(x0 + 1, w - 1)
                y0 = int(gy)
                y1 = min(y0 + 1, h - 1)

                wa = (x1 - gx) * (y1 - gy)
                wb = (gx - x0) * (y1 - gy)
                wc = (x1 - gx) * (gy - y0)
                wd = (gx - x0) * (gy - y0)

                for k in range(c):
                    a = img[y0, x0, k]
                    b = img[y0, x1, k]
                    c_ = img[y1, x0, k]
                    d = img[y1, x1, k]
                    value = wa * a + wb * b + wc * c_ + wd * d
                    resized[y, x, k] = np.clip(value, 0, 255)
        return resized

    def resize_average(self, img, factor):
        h, w, c = img.shape
        new_h = int(h * factor)
        new_w = int(w * factor)
        resized = np.zeros((new_h, new_w, c), dtype=np.uint8)
        for y in range(new_h):
            for x in range(new_w):
                src_x = x / factor
                src_y = y / factor
                x0 = int(src_x)
                y0 = int(src_y)
                x1 = min(x0 + 1, w - 1)
                y1 = min(y0 + 1, h - 1)
                region = img[y0:y1+1, x0:x1+1]
                if region.size > 0:
                    resized[y, x] = np.mean(region, axis=(0, 1))
        return resized

    def rotate_image(self):
        if self.displayed_image is None:
            return

        try:
            angle = float(self.angle_input.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Hata", "Geçerli bir açı girin.")
            return

        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        h, w, c = self.displayed_image.shape
        cx, cy = w // 2, h // 2


        new_w = int(abs(w * cos_a) + abs(h * sin_a))
        new_h = int(abs(h * cos_a) + abs(w * sin_a))

        rotated = np.zeros((new_h, new_w, c), dtype=np.uint8)
        ox, oy = new_w // 2, new_h // 2

        for y in range(new_h):
            for x in range(new_w):
                tx = x - ox
                ty = y - oy

                src_x = int(cos_a * tx + sin_a * ty + cx)
                src_y = int(-sin_a * tx + cos_a * ty + cy)

                if 0 <= src_x < w and 0 <= src_y < h:
                    rotated[y, x] = self.displayed_image[src_y, src_x]

        self.displayed_image = rotated
        self.display_image(rotated)

    def display_image(self, img):
        h, w, ch = img.shape
        bytes_per_line = 3 * w
        q_image = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(500, 300, QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)


class ExcelWriterThread(QThread):
    finished = pyqtSignal(str)
    display_image = pyqtSignal(np.ndarray)

    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def run(self):
        try:
            img = cv2.imread(self.file_name)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 200])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            data = []
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                cx, cy = centroids[i]
                roi = mask[y:y + h, x:x + w]
                energy = np.sum(roi.astype(np.float32) ** 2)
                entropy = -np.sum((roi / 255.0) * np.log2((roi / 255.0) + 1e-9))
                mean_val = np.mean(roi)
                median_val = np.median(roi)
                diagonal = int(np.sqrt(w ** 2 + h ** 2))
                data.append({
                    "Nesne No": i,
                    "Merkez (x,y)": f"({int(cx)}, {int(cy)})",
                    "Boyut (WxH)": f"{w}x{h}",
                    "Çapraz (Diagonal)": diagonal,
                    "Alan": area,
                    "Enerji": round(energy, 2),
                    "Entropi": round(entropy, 2),
                    "Ortalama": round(mean_val, 2),
                    "Medyan": round(median_val, 2)
                })

            df = pd.DataFrame(data)
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "nesnenin_ozellikleri.xlsx")
            df.to_excel(desktop_path, index=False)

            color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.display_image.emit(color_mask)
            self.finished.emit("Excel dosyası ve görsel başarıyla hazırlandı.")
        except Exception as e:
            self.finished.emit(f"Hata oluştu: {e}")


class FinalWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Final Penceresi")
        self.setGeometry(100, 100, 900, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        self.image_label = QLabel("Henüz resim eklenmedi.")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setMinimumSize(400, 300)
        main_layout.addWidget(self.image_label)

        btn_layout = QHBoxLayout()
        main_layout.addLayout(btn_layout)

        self.select_button = QPushButton("Resim Ekle")
        self.select_button.clicked.connect(self.load_image)
        btn_layout.addWidget(self.select_button)

        self.standard_btn = QPushButton("Standart Sigmoid")
        self.standard_btn.clicked.connect(self.apply_standard_sigmoid)
        self.standard_btn.setEnabled(False)
        btn_layout.addWidget(self.standard_btn)

        self.shifted_btn = QPushButton("Yatay Kaydırılmış Sigmoid")
        self.shifted_btn.clicked.connect(self.apply_shifted_sigmoid)
        self.shifted_btn.setEnabled(False)
        btn_layout.addWidget(self.shifted_btn)

        self.steep_btn = QPushButton("Eğimli Sigmoid")
        self.steep_btn.clicked.connect(self.apply_steep_sigmoid)
        self.steep_btn.setEnabled(False)
        btn_layout.addWidget(self.steep_btn)

        self.custom_btn = QPushButton("Kendi Fonksiyon")
        self.custom_btn.clicked.connect(self.apply_custom_function)
        self.custom_btn.setEnabled(False)
        btn_layout.addWidget(self.custom_btn)

        self.hough_lines_button = QPushButton("Yol Çizgilerini Tespit Et")
        self.hough_lines_button.clicked.connect(self.detect_hough_lines)
        btn_layout.addWidget(self.hough_lines_button)

        self.hough_eyes_button = QPushButton("Gözleri Tespit Et")
        self.hough_eyes_button.clicked.connect(self.detect_hough_eyes)
        btn_layout.addWidget(self.hough_eyes_button)

        self.deblur_button = QPushButton("Deblurring Uygula")
        self.deblur_button.clicked.connect(self.apply_deblurring)
        btn_layout.addWidget(self.deblur_button)

        self.object_stats_button = QPushButton("Nesne Say ve Özellikleri Çıkar")
        self.object_stats_button.clicked.connect(self.extract_object_features)
        btn_layout.addWidget(self.object_stats_button)

        self.original_pixmap = None
        self.gray_pixels = None

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            pixmap = QPixmap(file_name)
            if pixmap.isNull():
                QtWidgets.QMessageBox.warning(self, "Hata", "Resim dosyası yüklenemedi.")
                return
            self.original_pixmap = pixmap
            self.image_label.setPixmap(self.original_pixmap)
            qimage_gray = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
            self.gray_pixels = self.qimage_to_gray_array(qimage_gray)

            self.standard_btn.setEnabled(True)
            self.shifted_btn.setEnabled(True)
            self.steep_btn.setEnabled(True)
            self.custom_btn.setEnabled(True)

    def qimage_to_gray_array(self, qimage):
        width = qimage.width()
        height = qimage.height()
        return [[qimage.pixelColor(x, y).red() for x in range(width)] for y in range(height)]

    def array_to_qimage(self, pixels):
        height = len(pixels)
        width = len(pixels[0]) if height > 0 else 0
        img = QImage(width, height, QImage.Format_Grayscale8)
        for y in range(height):
            for x in range(width):
                val = max(0, min(255, int(pixels[y][x])))
                img.setPixelColor(x, y, QColor(val, val, val))
        return img

    def show_image(self, qimage):
        pix = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pix)

    def normalize(self, pixels):
        return [[v / 255.0 for v in row] for row in pixels]

    def denormalize(self, pixels):
        return [[min(255, max(0, int(v * 255))) for v in row] for row in pixels]

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def apply_function(self, func):
        norm = self.normalize(self.gray_pixels)
        result = [[func(norm[y][x]) for x in range(len(norm[0]))] for y in range(len(norm))]
        denorm = self.denormalize(result)
        qimg = self.array_to_qimage(denorm)
        self.show_image(qimg)

    def apply_standard_sigmoid(self):
        self.apply_function(lambda x: self.sigmoid(12 * x - 6))

    def apply_shifted_sigmoid(self):
        self.apply_function(lambda x: self.sigmoid(12 * (x - 0.5)))

    def apply_steep_sigmoid(self):
        self.apply_function(lambda x: self.sigmoid(20 * (x - 0.5)))

    def apply_custom_function(self):
        def f(x):
            base = self.sigmoid(15 * x - 7.5)
            sine = 0.2 * math.sin(math.pi * x)
            return max(0, min(1, base + sine))
        self.apply_function(f)

    def detect_hough_lines(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Yol Resmi Seç", "", "Image Files (*.png *.jpg *.bmp)")
        if not file_name:
            return
        image = cv2.imread(file_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
                x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.display_cv_image(image)

    def detect_hough_eyes(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Yüz Resmi Seç", "", "Image Files (*.png *.jpg *.bmp)")
        if not file_name:
            return
        image = cv2.imread(file_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 30,
                                   param1=50, param2=30, minRadius=10, maxRadius=40)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        self.display_cv_image(image)

    def apply_deblurring(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Bulanık Resim Seç", "", "Image Files (*.png *.jpg *.bmp)")
        if not file_name:
            return
        image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]]) / 256.0
        blurred = cv2.filter2D(gray, -1, kernel)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        sharpened_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        self.display_cv_image(sharpened_color)

    def extract_object_features(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görsel Seç (Yeşil Nesneler)", "",
                                                   "Image Files (*.png *.jpg *.bmp)")
        if not file_name:
            return

        self.writer_thread = ExcelWriterThread(file_name)
        self.writer_thread.finished.connect(lambda msg: QtWidgets.QMessageBox.information(self, "Bilgi", msg))
        self.writer_thread.display_image.connect(self.display_cv_image)
        self.writer_thread.start()

    def display_cv_image(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img).scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)





class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1068, 803)
        MainWindow.setStyleSheet("background-color: rgb(85, 170, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("221229037_Hasan_Ağaoğlu")
        self.label.setGeometry(QtCore.QRect(850, 700, 191, 51))
        self.label.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setText("Dijital Görüntü İşleme Arayüzü")
        self.label_2.setGeometry(QtCore.QRect(370, 0, 181, 51))
        self.label_2.setStyleSheet("background-color: rgb(0, 170, 255);")

        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 70, 378, 131))
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.pushButton = QtWidgets.QPushButton("Ödev1 Temel İşlevliği", self.horizontalLayoutWidget)
        self.pushButton.setStyleSheet("background-color: rgb(255, 85, 0);")
        self.horizontalLayout.addWidget(self.pushButton)

        self.pushButton_2 = QtWidgets.QPushButton("Ödev2 Filtre Uygulama", self.horizontalLayoutWidget)
        self.pushButton_2.setStyleSheet("background-color: rgb(255, 85, 0);")
        self.horizontalLayout.addWidget(self.pushButton_2)

        self.pushButton_3 = QtWidgets.QPushButton("Histogram", self.horizontalLayoutWidget)
        self.pushButton_3.setStyleSheet("background-color: rgb(0, 0, 255);")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.final_button = QtWidgets.QPushButton("Final", self.centralwidget)
        self.final_button.setGeometry(QtCore.QRect(100, 400, 200, 50))
        self.final_button.setStyleSheet("background-color: orange; font-size: 16px;")
        self.final_button.clicked.connect(self.open_final)

        MainWindow.setCentralWidget(self.centralwidget)
        self.pushButton.clicked.connect(self.open_odev1_window)
        self.pushButton_2.clicked.connect(self.open_odev2_window)
        self.pushButton_3.clicked.connect(self.open_histogram_window)

    def open_histogram_window(self):
        self.histogram_window = HistogramWindow()
        self.histogram_window.show()

    def open_odev1_window(self):
        self.odev1_window = Odev1Window()
        self.odev1_window.show()

    def open_odev2_window(self):
        self.odev2_window = Odev2Window()
        self.odev2_window.show()

    def open_final(self):
        self.final_window = FinalWindow()
        self.final_window.show()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
