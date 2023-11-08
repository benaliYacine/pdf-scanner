from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGraphicsView, QGraphicsScene, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import fitz  # PyMuPDF
from PIL import Image
from PyQt5.QtGui import QPainter, QTransform
from io import BytesIO
class ImprovedPDFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Improved PDF Viewer")
        self.zoom_factor = 1.0
        self.pdf = None
        self.page_images = []

        # Central Widget and Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Graphics View and Scene
        self.view = QGraphicsView(self)
        layout.addWidget(self.view)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        # Buttons
        self.load_button = QPushButton("Load PDF", self)
        layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.load_pdf)

        self.zoom_in_button = QPushButton("Zoom In", self)
        layout.addWidget(self.zoom_in_button)
        self.zoom_in_button.clicked.connect(self.zoom_in)

        self.zoom_out_button = QPushButton("Zoom Out", self)
        layout.addWidget(self.zoom_out_button)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.view.wheelEvent = self.on_mouse_scroll

    def on_mouse_scroll(self, event):
        # Check for Ctrl key being pressed for zooming with scroll
        if event.modifiers() == Qt.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:  # Zoom in
                self.zoom_in()
            else:  # Zoom out
                self.zoom_out()
        else:
            # Default behavior: scrolling
            super(QGraphicsView, self.view).wheelEvent(event)

    def load_pdf(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF files (*.pdf)")
            if file_path:
                self.pdf = fitz.open(file_path)
                print('1')
                self.render_pdf()
                print('3')
                self.display_pdf()
                print('9')
        except:
            print(f"mouchkil f load ")      
    def render_pdf(self):
        images = []
        zoom=200//72
        for page_num in range(len(self.pdf)):
            page = self.pdf[page_num]
            zoom_matrix = fitz.Matrix(zoom,zoom)
            pix = page.get_pixmap(matrix=zoom_matrix)
            fmt = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(fmt, [pix.width, pix.height], pix.samples)
            images.append(img)

        # Stitch the images vertically
        total_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        self.stitched_image = Image.new('RGB', (total_width, total_height))
        y_offset = 0
        for img in images:
            self.stitched_image.paste(img, (0, y_offset))
            y_offset += img.height

    def display_pdf(self):
        self.scene.clear()
        scaled_width = int(self.stitched_image.width * self.zoom_factor)
        scaled_height = int(self.stitched_image.height * self.zoom_factor)
        scaled_image = self.stitched_image.resize((scaled_width, scaled_height), Image.LANCZOS)

        # Convert the PIL image to JPEG format
        buffered = BytesIO()
        scaled_image.save(buffered, format="JPEG")
        jpeg_data = buffered.getvalue()

        # Convert the JPEG data to QImage
        qt_image = QImage.fromData(jpeg_data)

        if not qt_image.isNull():
            pixmap = QPixmap.fromImage(qt_image)
            self.scene.addPixmap(pixmap)
            self.view.setSceneRect(0, 0, scaled_width, scaled_height)
        else:
            print("Failed to convert JPEG data to QImage. QImage is null.")

    def zoom_in(self):
        self.zoom_factor = min(4.0, self.zoom_factor + 0.3)
        self.apply_zoom()

    def zoom_out(self):
        self.zoom_factor = max(0.4, self.zoom_factor - 0.3)
        self.apply_zoom()

    def apply_zoom(self):
        transform = QTransform()
        transform.scale(self.zoom_factor, self.zoom_factor)
        self.view.setTransform(transform)

# The following code block will not run here due to the GUI nature of the app, but it can be used outside for testing
if __name__ == "__main__":
    app = QApplication([])
    viewer = ImprovedPDFViewer()
    viewer.show()
    app.exec_()
