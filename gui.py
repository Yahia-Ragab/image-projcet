import sys
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QHBoxLayout, QComboBox, QFrame, QLineEdit, QTextEdit, QCheckBox, QSlider
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import cv2
from filter import Filter
from transformation import Transformation
from resize import Resize
from info import Info
from analysis import Analysis
from compress import Compress

class ImageDropWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Image GUI")
        self.resize(1000, 600)
        self.setAcceptDrops(True)

        self.current_image_path = None
        self.filtered_img = None
        self.transformed_img = None
        self.resized_img = None
        self.compressed_data = None
        self.compression_method = None
        
        self.last_filter_choice = "None"
        self.last_filter_params = {}
        self.last_resize_choice = "None"
        self.last_resize_params = {}
        self.last_trans_choice = "None"
        self.last_trans_params = {}
        
        self.image_width = 0
        self.image_height = 0

        self.menu_layout = QVBoxLayout()

        # Reset Button
        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.clicked.connect(self.reset_all)
        self.reset_btn.setStyleSheet("background-color: #ff4444; color: white; font-weight: bold; padding: 8px;")
        self.menu_layout.addWidget(self.reset_btn)

        # Grayscale Checkbox
        self.grayscale_checkbox = QCheckBox("Apply Grayscale")
        self.grayscale_checkbox.stateChanged.connect(self.apply_grayscale)
        self.menu_layout.addWidget(self.grayscale_checkbox)

        self.filter_label = QLabel("Filters")
        self.menu_layout.addWidget(self.filter_label)

        self.combo1 = QComboBox()
        self.combo1.addItems([
            "None", "Blur", "Median", "Laplacian", "Sobel",
            "Gradient", "Sharpen", "Binary", "Adjust"
        ])
        self.combo1.currentTextChanged.connect(self.show_filter_inputs)
        self.menu_layout.addWidget(self.combo1)

        self.filter_input1 = QLineEdit()
        self.filter_input1.setPlaceholderText("Param 1")
        self.filter_input2 = QLineEdit()
        self.filter_input2.setPlaceholderText("Param 2")
        self.filter_input3 = QLineEdit()
        self.filter_input3.setPlaceholderText("Param 3")

        for widget in (self.filter_input1, self.filter_input2, self.filter_input3):
            widget.hide()
            self.menu_layout.addWidget(widget)

        self.apply_filter_btn = QPushButton("Apply Filter")
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        self.menu_layout.addWidget(self.apply_filter_btn)

        self.resize_label = QLabel("Resize")
        self.menu_layout.addWidget(self.resize_label)

        self.combo_resize = QComboBox()
        self.combo_resize.addItems([
            "None", "Default", "Nearest Neighbor", "Bilinear", "Bicubic"
        ])
        self.combo_resize.currentTextChanged.connect(self.show_resize_inputs)
        self.menu_layout.addWidget(self.combo_resize)

        self.resize_input1 = QLineEdit()
        self.resize_input1.setPlaceholderText("Width")
        self.resize_input2 = QLineEdit()
        self.resize_input2.setPlaceholderText("Height")

        for widget in (self.resize_input1, self.resize_input2):
            widget.hide()
            self.menu_layout.addWidget(widget)

        self.apply_resize_btn = QPushButton("Apply Resize")
        self.apply_resize_btn.clicked.connect(self.apply_resize)
        self.menu_layout.addWidget(self.apply_resize_btn)

        self.trans_label = QLabel("Transformations")
        self.menu_layout.addWidget(self.trans_label)

        self.combo_trans = QComboBox()
        self.combo_trans.addItems([
            "None", "Rotate", "Crop", "Shear X", "Shear Y", "Translate"
        ])
        self.combo_trans.currentTextChanged.connect(self.show_trans_inputs)
        self.menu_layout.addWidget(self.combo_trans)

        # Transformation sliders
        self.trans_slider1_label = QLabel("")
        self.trans_slider1 = QSlider(Qt.Orientation.Horizontal)
        self.trans_slider1.valueChanged.connect(self.update_trans_slider_labels)
        
        self.trans_slider2_label = QLabel("")
        self.trans_slider2 = QSlider(Qt.Orientation.Horizontal)
        self.trans_slider2.valueChanged.connect(self.update_trans_slider_labels)
        
        self.trans_slider3_label = QLabel("")
        self.trans_slider3 = QSlider(Qt.Orientation.Horizontal)
        self.trans_slider3.valueChanged.connect(self.update_trans_slider_labels)
        
        self.trans_slider4_label = QLabel("")
        self.trans_slider4 = QSlider(Qt.Orientation.Horizontal)
        self.trans_slider4.valueChanged.connect(self.update_trans_slider_labels)

        for label, slider in [(self.trans_slider1_label, self.trans_slider1),
                              (self.trans_slider2_label, self.trans_slider2),
                              (self.trans_slider3_label, self.trans_slider3),
                              (self.trans_slider4_label, self.trans_slider4)]:
            label.hide()
            slider.hide()
            self.menu_layout.addWidget(label)
            self.menu_layout.addWidget(slider)

        self.apply_transform_btn = QPushButton("Apply Transformation")
        self.apply_transform_btn.clicked.connect(self.apply_transformation)
        self.menu_layout.addWidget(self.apply_transform_btn)

        self.analysis_label = QLabel("Analysis")
        self.menu_layout.addWidget(self.analysis_label)

        self.combo_analysis = QComboBox()
        self.combo_analysis.addItems([
            "None", "Threshold Analysis", "Histogram"
        ])
        self.combo_analysis.currentTextChanged.connect(self.perform_analysis)
        self.menu_layout.addWidget(self.combo_analysis)

        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setMaximumHeight(200)
        self.info_display.setPlaceholderText("Image info will appear here")
        self.menu_layout.addWidget(self.info_display)

        self.compress_label = QLabel("Compression")
        self.menu_layout.addWidget(self.compress_label)

        self.combo_compress = QComboBox()
        self.combo_compress.addItems([
            "None", "Huffman", "Golomb-Rice", "Arithmetic", "LZW", 
            "RLE", "Symbol-Based", "Bit-Plane", "DCT", "Predictive", "Wavelet"
        ])
        self.combo_compress.currentTextChanged.connect(self.show_compress_inputs)
        self.menu_layout.addWidget(self.combo_compress)

        self.compress_input1 = QLineEdit()
        self.compress_input1.setPlaceholderText("Param 1")
        self.compress_input1.hide()
        self.menu_layout.addWidget(self.compress_input1)
        
        self.compress_input2 = QLineEdit()
        self.compress_input2.setPlaceholderText("Quality/Threshold")
        self.compress_input2.hide()
        self.menu_layout.addWidget(self.compress_input2)

        self.apply_compress_btn = QPushButton("Apply Compression")
        self.apply_compress_btn.clicked.connect(self.apply_compression)
        self.menu_layout.addWidget(self.apply_compress_btn)

        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        self.menu_layout.addWidget(self.save_btn)
        self.menu_layout.addStretch()

        self.drop_label = QLabel("Drag an image or Browse")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setFrameShape(QFrame.Shape.Box)

        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFrameShape(QFrame.Shape.Box)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.open_file)

        right_layout = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(self.drop_label)
        top.addWidget(self.preview_label)
        right_layout.addLayout(top)
        right_layout.addWidget(self.browse_btn)

        main = QHBoxLayout()
        main.addLayout(self.menu_layout, 1)
        main.addLayout(right_layout, 3)
        self.setLayout(main)

    def reset_all(self):
        """Reset all settings but keep the original image displayed"""
        if not self.current_image_path:
            return
            
        # Reset processing results
        self.transformed_img = None
        self.resized_img = None
        self.compressed_data = None
        self.compression_method = None
        
        self.last_filter_choice = "None"
        self.last_filter_params = {}
        self.last_resize_choice = "None"
        self.last_resize_params = {}
        self.last_trans_choice = "None"
        self.last_trans_params = {}
        
        # Reset UI elements
        self.grayscale_checkbox.setChecked(False)
        self.combo1.setCurrentText("None")
        self.combo_resize.setCurrentText("None")
        self.combo_trans.setCurrentText("None")
        self.combo_analysis.setCurrentText("None")
        self.combo_compress.setCurrentText("None")
        
        # Reset to original image without reloading
        f = Filter(self.current_image_path)
        self.filtered_img = f.img
        
        # Update preview to show original
        self.update_preview()
        self.update_info()

    def apply_grayscale(self, state):
        """Apply or remove grayscale filter"""
        if not self.current_image_path:
            return
            
        if state == Qt.CheckState.Checked.value:
            f = Filter(self.current_image_path)
            self.filtered_img = f.to_gray()
        else:
            f = Filter(self.current_image_path)
            self.filtered_img = f.img
            
        self.reapply_pipeline_from_filter()

    def update_trans_slider_labels(self):
        """Update slider value labels"""
        choice = self.combo_trans.currentText()
        
        if choice == "Rotate":
            self.trans_slider1_label.setText(f"Angle: {self.trans_slider1.value()}°")
        elif choice == "Crop":
            self.trans_slider1_label.setText(f"X: {self.trans_slider1.value()}")
            self.trans_slider2_label.setText(f"Y: {self.trans_slider2.value()}")
            self.trans_slider3_label.setText(f"Width: {self.trans_slider3.value()}")
            self.trans_slider4_label.setText(f"Height: {self.trans_slider4.value()}")
        elif choice in ("Shear X", "Shear Y"):
            self.trans_slider1_label.setText(f"Factor: {self.trans_slider1.value() / 100.0:.2f}")
        elif choice == "Translate":
            self.trans_slider1_label.setText(f"TX: {self.trans_slider1.value()}")
            self.trans_slider2_label.setText(f"TY: {self.trans_slider2.value()}")

    def cv_to_pixmap(self, img):
        if img is None:
            return QPixmap()
        if len(img.shape) == 2:
            qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            self.load_image(url.toLocalFile())
            break

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        self.current_image_path = path
        pix = QPixmap(path)
        if not pix.isNull():
            # Store image dimensions
            img = cv2.imread(path)
            if img is not None:
                self.image_height, self.image_width = img.shape[:2]
            
            scaled1 = pix.scaled(self.drop_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            scaled2 = pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio)

            self.drop_label.setPixmap(scaled1)
            self.preview_label.setPixmap(scaled2)

            # Don't reset filtered_img if we're just resetting settings
            if self.filtered_img is None:
                # Apply grayscale if checkbox is checked
                if self.grayscale_checkbox.isChecked():
                    f = Filter(path)
                    self.filtered_img = f.to_gray()
                else:
                    f = Filter(path)
                    self.filtered_img = f.img

            self.update_info()
            self.show_filter_inputs()

    def show_filter_inputs(self):
        for w in (self.filter_input1, self.filter_input2, self.filter_input3):
            w.hide()

        choice = self.combo1.currentText()

        if choice == "Blur":
            self.filter_input1.setPlaceholderText("Kernel size (odd, e.g., 19)")
            self.filter_input2.setPlaceholderText("Sigma (e.g., 3)")
            self.filter_input1.show()
            self.filter_input2.show()
        elif choice == "Median":
            self.filter_input1.setPlaceholderText("Kernel size (odd, e.g., 7)")
            self.filter_input1.show()
        elif choice == "Laplacian":
            self.filter_input1.setPlaceholderText("Kernel size (odd, e.g., 3)")
            self.filter_input1.show()
        elif choice == "Sobel":
            self.filter_input1.setPlaceholderText("Kernel size (odd, e.g., 3)")
            self.filter_input1.show()
        elif choice == "Adjust":
            self.filter_input1.setPlaceholderText("Brightness (e.g., 0)")
            self.filter_input2.setPlaceholderText("Contrast (e.g., 0)")
            self.filter_input1.show()
            self.filter_input2.show()

        if choice in ["None", "Gradient", "Sharpen", "Binary"]:
            self.apply_filter()

    def apply_filter(self):
        if not self.current_image_path:
            return

        # Get base image (with or without grayscale)
        if self.grayscale_checkbox.isChecked():
            f = Filter(self.current_image_path)
            base_img = f.to_gray()
            temp_path = "/tmp/temp_gray.png"
            cv2.imwrite(temp_path, base_img)
            f = Filter(temp_path)
        else:
            f = Filter(self.current_image_path)

        choice = self.combo1.currentText()

        try:
            if choice == "Blur":
                k = int(self.filter_input1.text()) if self.filter_input1.text() else 19
                sigma = int(self.filter_input2.text()) if self.filter_input2.text() else 3
                img = f.to_blur(k, sigma)
                self.last_filter_params = {'k': k, 'sigma': sigma}
            elif choice == "Median":
                k = int(self.filter_input1.text()) if self.filter_input1.text() else 7
                img = f.median(k)
                self.last_filter_params = {'k': k}
            elif choice == "Laplacian":
                k = int(self.filter_input1.text()) if self.filter_input1.text() else 3
                img = f.laplacian(k)
                self.last_filter_params = {'k': k}
            elif choice == "Sobel":
                k = int(self.filter_input1.text()) if self.filter_input1.text() else 3
                img = f.sobel(k)
                self.last_filter_params = {'k': k}
            elif choice == "Gradient":
                img = f.gradient()
            elif choice == "Sharpen":
                img = f.sharpen()
            elif choice == "Binary":
                img, _ = f.to_binary()
            elif choice == "Adjust":
                brightness = int(self.filter_input1.text()) if self.filter_input1.text() else 0
                contrast = int(self.filter_input2.text()) if self.filter_input2.text() else 0
                img = f.adjust(brightness, contrast)
                self.last_filter_params = {'brightness': brightness, 'contrast': contrast}
            else:
                img = f.img
        except:
            img = f.img

        self.filtered_img = img
        self.last_filter_choice = choice
        
        self.reapply_pipeline_from_filter()

    def reapply_pipeline_from_filter(self):
        if self.last_resize_choice != "None":
            self.reapply_resize()
        else:
            self.resized_img = None
            
        if self.last_trans_choice != "None":
            self.reapply_transformation()
        else:
            self.transformed_img = None
            
        self.update_preview()
        self.update_info()

    def reapply_resize(self):
        if self.filtered_img is None or self.last_resize_choice == "None":
            return
            
        temp_path = "/tmp/temp_for_resize.png"
        cv2.imwrite(temp_path, self.filtered_img)

        r = Resize(temp_path)
        
        try:
            width = self.last_resize_params.get('width')
            height = self.last_resize_params.get('height')
            
            if width and height:
                if self.last_resize_choice == "Default":
                    self.resized_img = r.resize(width, height)
                elif self.last_resize_choice == "Nearest Neighbor":
                    self.resized_img = r.resize_nn(width, height)
                elif self.last_resize_choice == "Bilinear":
                    self.resized_img = r.resize_bilinear(width, height)
                elif self.last_resize_choice == "Bicubic":
                    self.resized_img = r.resize_bicubic(width, height)
        except:
            pass

    def reapply_transformation(self):
        base_img = self.resized_img if self.resized_img is not None else self.filtered_img
        if base_img is None or self.last_trans_choice == "None":
            return

        temp_path = "/tmp/temp_filtered.png"
        cv2.imwrite(temp_path, base_img)

        t = Transformation(temp_path)
        
        try:
            if self.last_trans_choice == "Rotate":
                angle = self.last_trans_params.get('angle')
                if angle is not None:
                    self.transformed_img = t.rotate(angle)
            elif self.last_trans_choice == "Crop":
                x = self.last_trans_params.get('x')
                y = self.last_trans_params.get('y')
                w = self.last_trans_params.get('w')
                h = self.last_trans_params.get('h')
                if all(v is not None for v in [x, y, w, h]):
                    self.transformed_img = t.crop(x, y, w, h)
            elif self.last_trans_choice == "Shear X":
                f = self.last_trans_params.get('factor')
                if f is not None:
                    self.transformed_img = t.shear_x(f)
            elif self.last_trans_choice == "Shear Y":
                f = self.last_trans_params.get('factor')
                if f is not None:
                    self.transformed_img = t.shear_y(f)
            elif self.last_trans_choice == "Translate":
                tx = self.last_trans_params.get('tx')
                ty = self.last_trans_params.get('ty')
                if tx is not None and ty is not None:
                    self.transformed_img = t.translate(tx, ty)
        except:
            pass

    def update_preview(self):
        img = self.transformed_img if self.transformed_img is not None else (
            self.resized_img if self.resized_img is not None else self.filtered_img
        )
        if img is not None:
            pix = self.cv_to_pixmap(img)
            self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def show_resize_inputs(self):
        for w in (self.resize_input1, self.resize_input2):
            w.hide()

        choice = self.combo_resize.currentText()

        if choice in ["Default", "Nearest Neighbor", "Bilinear", "Bicubic"]:
            self.resize_input1.setPlaceholderText("Width")
            self.resize_input2.setPlaceholderText("Height")
            self.resize_input1.show()
            self.resize_input2.show()

    def apply_resize(self):
        base_img = self.filtered_img if self.filtered_img is not None else None
        if not self.current_image_path or base_img is None:
            return

        temp_path = "/tmp/temp_for_resize.png"
        cv2.imwrite(temp_path, base_img)

        r = Resize(temp_path)
        choice = self.combo_resize.currentText()

        try:
            if choice in ["Default", "Nearest Neighbor", "Bilinear", "Bicubic"]:
                width = int(self.resize_input1.text())
                height = int(self.resize_input2.text())
                
                self.last_resize_choice = choice
                self.last_resize_params = {'width': width, 'height': height}

                if choice == "Default":
                    img = r.resize(width, height)
                elif choice == "Nearest Neighbor":
                    img = r.resize_nn(width, height)
                elif choice == "Bilinear":
                    img = r.resize_bilinear(width, height)
                elif choice == "Bicubic":
                    img = r.resize_bicubic(width, height)
            else:
                return
        except:
            return

        self.resized_img = img
        
        if self.last_trans_choice != "None":
            self.reapply_transformation()
        else:
            self.transformed_img = None
            
        self.update_preview()
        self.update_info()

    def show_trans_inputs(self):
        # Hide all sliders
        for label, slider in [(self.trans_slider1_label, self.trans_slider1),
                              (self.trans_slider2_label, self.trans_slider2),
                              (self.trans_slider3_label, self.trans_slider3),
                              (self.trans_slider4_label, self.trans_slider4)]:
            label.hide()
            slider.hide()

        choice = self.combo_trans.currentText()

        if choice == "Rotate":
            self.trans_slider1_label.setText(f"Angle: 0°")
            self.trans_slider1.setMinimum(0)
            self.trans_slider1.setMaximum(360)
            self.trans_slider1.setValue(0)
            self.trans_slider1_label.show()
            self.trans_slider1.show()
            
        elif choice == "Crop":
            self.trans_slider1_label.setText(f"X: 0")
            self.trans_slider1.setMinimum(0)
            self.trans_slider1.setMaximum(self.image_width)
            self.trans_slider1.setValue(0)
            
            self.trans_slider2_label.setText(f"Y: 0")
            self.trans_slider2.setMinimum(0)
            self.trans_slider2.setMaximum(self.image_height)
            self.trans_slider2.setValue(0)
            
            self.trans_slider3_label.setText(f"Width: {self.image_width}")
            self.trans_slider3.setMinimum(1)
            self.trans_slider3.setMaximum(self.image_width)
            self.trans_slider3.setValue(self.image_width)
            
            self.trans_slider4_label.setText(f"Height: {self.image_height}")
            self.trans_slider4.setMinimum(1)
            self.trans_slider4.setMaximum(self.image_height)
            self.trans_slider4.setValue(self.image_height)
            
            for label, slider in [(self.trans_slider1_label, self.trans_slider1),
                                  (self.trans_slider2_label, self.trans_slider2),
                                  (self.trans_slider3_label, self.trans_slider3),
                                  (self.trans_slider4_label, self.trans_slider4)]:
                label.show()
                slider.show()
                
        elif choice in ("Shear X", "Shear Y"):
            self.trans_slider1_label.setText(f"Factor: 0.00")
            self.trans_slider1.setMinimum(-100)
            self.trans_slider1.setMaximum(100)
            self.trans_slider1.setValue(0)
            self.trans_slider1_label.show()
            self.trans_slider1.show()
            
        elif choice == "Translate":
            self.trans_slider1_label.setText(f"TX: 0")
            self.trans_slider1.setMinimum(-self.image_width)
            self.trans_slider1.setMaximum(self.image_width)
            self.trans_slider1.setValue(0)
            
            self.trans_slider2_label.setText(f"TY: 0")
            self.trans_slider2.setMinimum(-self.image_height)
            self.trans_slider2.setMaximum(self.image_height)
            self.trans_slider2.setValue(0)
            
            self.trans_slider1_label.show()
            self.trans_slider1.show()
            self.trans_slider2_label.show()
            self.trans_slider2.show()

    def apply_transformation(self):
        base_img = self.resized_img if self.resized_img is not None else self.filtered_img
        if base_img is None:
            return

        temp_path = "/tmp/temp_filtered.png"
        cv2.imwrite(temp_path, base_img)

        t = Transformation(temp_path)
        c = self.combo_trans.currentText()

        try:
            if c == "Rotate":
                angle = float(self.trans_slider1.value())
                img = t.rotate(angle)
                self.last_trans_choice = c
                self.last_trans_params = {'angle': angle}
            elif c == "Crop":
                x = int(self.trans_slider1.value())
                y = int(self.trans_slider2.value())
                w = int(self.trans_slider3.value())
                h = int(self.trans_slider4.value())
                img = t.crop(x, y, w, h)
                self.last_trans_choice = c
                self.last_trans_params = {'x': x, 'y': y, 'w': w, 'h': h}
            elif c == "Shear X":
                f = float(self.trans_slider1.value() / 100.0)
                img = t.shear_x(f)
                self.last_trans_choice = c
                self.last_trans_params = {'factor': f}
            elif c == "Shear Y":
                f = float(self.trans_slider1.value() / 100.0)
                img = t.shear_y(f)
                self.last_trans_choice = c
                self.last_trans_params = {'factor': f}
            elif c == "Translate":
                tx = int(self.trans_slider1.value())
                ty = int(self.trans_slider2.value())
                img = t.translate(tx, ty)
                self.last_trans_choice = c
                self.last_trans_params = {'tx': tx, 'ty': ty}
            else:
                return
        except:
            return

        self.transformed_img = img
        self.update_preview()
        self.update_info()

    def perform_analysis(self):
        if not self.current_image_path:
            return

        choice = self.combo_analysis.currentText()
        
        if choice == "None":
            self.update_info()
            return

        current_img = self.transformed_img if self.transformed_img is not None else (
            self.resized_img if self.resized_img is not None else (
                self.filtered_img if self.filtered_img is not None else None
            )
        )

        if current_img is None:
            temp_path = self.current_image_path
        else:
            temp_path = "/tmp/temp_analysis.png"
            cv2.imwrite(temp_path, current_img)

        try:
            a = Analysis(temp_path)
            info = Info(temp_path)
            width, height = info.get_resolution()
            size = info.get_size()
            file_type = info.get_type()
            channels = info.get_channel()

            info_text = f"""IMAGE INFORMATION
━━━━━━━━━━━━━━━━━━━━
Resolution: {width} x {height}
File Size: {size} MB
Type: {file_type}
Channels: {channels}

"""

            if choice == "Threshold Analysis":
                result = a.compute_threshold()
                info_text += f"""THRESHOLD ANALYSIS
━━━━━━━━━━━━━━━━━━━━
Average Threshold: {result['average_threshold']:.2f}
Otsu Threshold: {result['otsu_threshold']:.2f}
Difference: {result['difference']:.2f}
Optimal: {"Yes" if result['is_optimal'] else "No"}
"""
            elif choice == "Histogram":
                hist = a.compute_histogram()
                mean_val = hist.mean()
                max_val = hist.max()
                min_val = hist.min()
                info_text += f"""HISTOGRAM ANALYSIS
━━━━━━━━━━━━━━━━━━━━
Mean: {mean_val:.2f}
Max: {max_val:.2f}
Min: {min_val:.2f}
Total Bins: 256
"""

            self.info_display.setText(info_text)
        except Exception as e:
            self.info_display.setText(f"Analysis error: {str(e)}")

    def update_info(self):
        if not self.current_image_path:
            return

        current_img = self.transformed_img if self.transformed_img is not None else (
            self.resized_img if self.resized_img is not None else (
                self.filtered_img if self.filtered_img is not None else None
            )
        )

        if current_img is None:
            temp_path = self.current_image_path
        else:
            temp_path = "/tmp/temp_info.png"
            cv2.imwrite(temp_path, current_img)

        try:
            info = Info(temp_path)
            width, height = info.get_resolution()
            size = info.get_size()
            file_type = info.get_type()
            channels = info.get_channel()

            info_text = f"""IMAGE INFORMATION
━━━━━━━━━━━━━━━━━━━━
Resolution: {width} x {height}
File Size: {size} MB
Type: {file_type}
Channels: {channels}
"""
            self.info_display.setText(info_text)
        except:
            pass

    def show_compress_inputs(self):
        self.compress_input1.hide()
        self.compress_input2.hide()
        
        choice = self.combo_compress.currentText()
        
        if choice == "Golomb-Rice":
            self.compress_input1.setPlaceholderText("M (power of 2, e.g., 4)")
            self.compress_input1.show()
        elif choice == "DCT":
            self.compress_input1.setPlaceholderText("Block size (e.g., 8)")
            self.compress_input2.setPlaceholderText("Quality (1-100, default 50)")
            self.compress_input1.show()
            self.compress_input2.show()
        elif choice == "Predictive":
            self.compress_input1.setPlaceholderText("Mode: left/top/avg")
            self.compress_input1.show()
        elif choice == "Wavelet":
            self.compress_input1.setPlaceholderText("Level (e.g., 1)")
            self.compress_input2.setPlaceholderText("Threshold (e.g., 10)")
            self.compress_input1.show()
            self.compress_input2.show()

    def apply_compression(self):
        current_img = self.transformed_img if self.transformed_img is not None else (
            self.resized_img if self.resized_img is not None else (
                self.filtered_img if self.filtered_img is not None else None
            )
        )
        
        if current_img is None and not self.current_image_path:
            return
            
        if current_img is None:
            temp_path = self.current_image_path
        else:
            temp_path = "/tmp/temp_compress.png"
            cv2.imwrite(temp_path, current_img)
        
        try:
            comp = Compress(temp_path)
            choice = self.combo_compress.currentText()
            
            if choice == "Huffman":
                encoded, huff_map = comp.huffman()
                stats = comp.get_compression_stats(encoded)
                self.compressed_data = (encoded, huff_map)
                self.compression_method = "huffman"
                preview = encoded[:200] + "..." if len(encoded) > 200 else encoded
                self.info_display.setText(
                    f"COMPRESSION: Huffman\nOriginal Bits: {stats['original_bits']}\n"
                    f"Compressed Bits: {stats['compressed_bits']}\n"
                    f"Compression Ratio: {stats['compression_ratio']}:1\n"
                    f"Space Saving: {stats['space_saving']}%\nHuffman Codes (sample):\n"
                    f"{str(dict(list(huff_map.items())[:10]))}\nEncoded Data (preview): {preview}"
                )
                
            elif choice == "Golomb-Rice":
                M = int(self.compress_input1.text()) if self.compress_input1.text() else 4
                encoded = comp.golomb_rice(M)
                encoded_str = "".join(encoded)
                stats = comp.get_compression_stats(encoded_str)
                self.compressed_data = encoded
                self.compression_method = "golomb"
                preview = encoded_str[:200] + "..." if len(encoded_str) > 200 else encoded_str
                self.info_display.setText(
                    f"COMPRESSION: Golomb-Rice\nM: {M}\nOriginal Bits: {stats['original_bits']}\n"
                    f"Compressed Bits: {stats['compressed_bits']}\nCompression Ratio: {stats['compression_ratio']}:1\n"
                    f"Space Saving: {stats['space_saving']}%\nEncoded Data (preview): {preview}"
                )
                
            elif choice == "Arithmetic":
                encoded, cum = comp.arithmetic()
                self.compressed_data = (encoded, cum)
                self.compression_method = "arithmetic"
                self.info_display.setText(
                    f"COMPRESSION: Arithmetic\nEncoded Value: {encoded:.15f}\n"
                    f"Sample Probability Ranges: {str(dict(list(cum.items())[:10]))}"
                )
                
            elif choice == "LZW":
                result, dict_size = comp.lzw()
                stats = comp.get_compression_stats(result)
                self.compressed_data = (result, dict_size)
                self.compression_method = "lzw"
                preview = str(result[:50]) + "..." if len(result) > 50 else str(result)
                self.info_display.setText(
                    f"COMPRESSION: LZW\nOriginal Bits: {stats['original_bits']}\n"
                    f"Compressed Bits: {stats['compressed_bits']}\nCompression Ratio: {stats['compression_ratio']}:1\n"
                    f"Space Saving: {stats['space_saving']}%\nEncoded Indices (preview): {preview}"
                )
                
            elif choice == "RLE":
                encoded = comp.rle()
                stats = comp.get_compression_stats(encoded)
                self.compressed_data = encoded
                self.compression_method = "rle"
                preview = str(encoded[:20]) + "..." if len(encoded) > 20 else str(encoded)
                self.info_display.setText(
                    f"COMPRESSION: RLE\nOriginal Bits: {stats['original_bits']}\n"
                    f"Compressed Bits: {stats['compressed_bits']}\nCompression Ratio: {stats['compression_ratio']}:1\n"
                    f"Space Saving: {stats['space_saving']}%\nRun-Length Pairs (preview): {preview}"
                )
                
            elif choice == "Symbol-Based":
                encoded, symbol_map = comp.symbol_based()
                stats = comp.get_compression_stats(encoded)
                self.compressed_data = (encoded, symbol_map)
                self.compression_method = "symbol"
                preview = str(encoded[:100]) + "..." if len(encoded) > 100 else str(encoded)
                self.info_display.setText(
                    f"COMPRESSION: Symbol-Based\nOriginal Bits: {stats['original_bits']}\n"
                    f"Compressed Bits: {stats['compressed_bits']}\nCompression Ratio: {stats['compression_ratio']}:1\n"
                    f"Space Saving: {stats['space_saving']}%\nSymbol Map Sample: {str(dict(list(symbol_map.items())[:10]))}\n"
                    f"Encoded Data (preview): {preview}"
                )
                
            elif choice == "Bit-Plane":
                bit_planes = comp.bit_plane()
                self.compressed_data = bit_planes
                self.compression_method = "bitplane"
                self.preview_label.setPixmap(
                    self.cv_to_pixmap(bit_planes[0]*255).scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
                )
                self.info_display.setText(f"COMPRESSION: Bit-Plane\nPlanes Generated: {len(bit_planes)}\nPreview shows Plane 0")
                
            elif choice == "DCT":
                block_size = int(self.compress_input1.text()) if self.compress_input1.text() else 8
                dct_result = comp.dct_blocks(block_size)
                self.compressed_data = dct_result
                self.compression_method = "dct"
                dct_display = np.uint8(np.clip(np.abs(dct_result), 0, 255))
                self.preview_label.setPixmap(
                    self.cv_to_pixmap(dct_display).scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
                )
                self.info_display.setText(f"COMPRESSION: DCT\nBlock Size: {block_size}x{block_size}\nOutput Shape: {dct_result.shape}")
                
            elif choice == "Predictive":
                mode = self.compress_input1.text() if self.compress_input1.text() else 'left'
                residual, predicted = comp.predictive(mode)
                self.compressed_data = (residual, predicted)
                self.compression_method = "predictive"
                residual_display = np.uint8(np.clip(residual+128, 0, 255))
                self.preview_label.setPixmap(
                    self.cv_to_pixmap(residual_display).scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
                )
                self.info_display.setText(f"COMPRESSION: Predictive\nMode: {mode}\nResidual Min: {residual.min()} Max: {residual.max()}")
                
            elif choice == "Wavelet":
                level = int(self.compress_input1.text()) if self.compress_input1.text() else 1
                wavelet_result = comp.wavelet(level)
                self.compressed_data = wavelet_result
                self.compression_method = "wavelet"
                wavelet_display = np.uint8(np.clip(np.abs(wavelet_result), 0, 255))
                self.preview_label.setPixmap(
                    self.cv_to_pixmap(wavelet_display).scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
                )
                self.info_display.setText(f"COMPRESSION: Wavelet\nLevel: {level}\nOutput Shape: {wavelet_result.shape}")
                
        except Exception as e:
            self.info_display.setText(f"Compression error: {str(e)}")


    def save_image(self):
        if not self.current_image_path:
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Save Image", 
            "", 
            "JPG (*.jpg);;NumPy Array (*.npy)"
        )
        if not path:
            return

        current_img = self.transformed_img or self.resized_img or self.filtered_img

        if path.endswith('.npy'):
            if self.compressed_data is not None:
                if isinstance(self.compressed_data, tuple):
                    np.save(path, self.compressed_data[0])
                else:
                    np.save(path, self.compressed_data)
            elif current_img is not None:
                np.save(path, current_img)
        else:
            if not path.lower().endswith('.jpg'):
                path += '.jpg'
            if self.compressed_data is not None:
                if self.compression_method in ["bitplane", "predictive", "wavelet", "dct"]:
                    img_to_save = np.uint8(np.clip(self.compressed_data, 0, 255))
                else:
                    img_to_save = current_img
            else:
                img_to_save = current_img

            if img_to_save is not None:
                import cv2
                cv2.imwrite(path, img_to_save)

app = QApplication(sys.argv)
window = ImageDropWindow()
window.show()
sys.exit(app.exec())