import tkinter as tk
from tkinter import filedialog
import fitz  # PyMuPDF
from PIL import Image, ImageTk

class PDFViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple PDF Viewer")

        self.zoom_factor = 1.0
        self.base_zoom = 1.0  # initial rendering resolution
        self.pdf = None
        self.stitched_image = None
        self.tk_image = None

        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=1)

        # Adding Scrollbars to the frame
        self.v_scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar = tk.Scrollbar(self.frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Adding Canvas to the frame with scrollbar command
        self.canvas = tk.Canvas(self.frame, bg="gray",
                                xscrollcommand=self.h_scrollbar.set,
                                yscrollcommand=self.v_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Configuring the scrollbars
        self.v_scrollbar.config(command=self.canvas.yview)
        self.h_scrollbar.config(command=self.canvas.xview)

        # Mouse drag bindings
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)

        self.load_button = tk.Button(self.root, text="Load PDF", command=self.load_pdf)
        self.load_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.zoom_in_button = tk.Button(self.root, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.zoom_out_button = tk.Button(self.root, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.bind("<MouseWheel>", self.on_mouse_scroll) 


    def on_mouse_scroll(self, event):
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")  # Scroll up
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")

    def on_mouse_down(self, event):
        # Mark the start point of drag
        self.canvas.scan_mark(event.x, event.y)

    def on_mouse_drag(self, event):
        # Adjust the canvas view based on the drag
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def load_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.pdf = fitz.open(file_path)
            self.render_pdf()
            self.display_pdf()

    def render_pdf(self):
        images = []

        for page_num in range(len(self.pdf)):
            page = self.pdf[page_num]
            zoom_matrix = fitz.Matrix(self.base_zoom, self.base_zoom)
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
        self.canvas.delete("all")  # clear the canvas for new PDF
        scaled_width = int(self.stitched_image.width * self.zoom_factor)
        scaled_height = int(self.stitched_image.height * self.zoom_factor)
        scaled_image = self.stitched_image.resize((scaled_width, scaled_height), Image.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(scaled_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.image = self.tk_image  # Keep a reference to avoid garbage collection
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def zoom_in(self):
        self.zoom_factor = min(4.0, self.zoom_factor + 0.1)
        self.display_pdf()

    def zoom_out(self):
        self.zoom_factor = max(0.5, self.zoom_factor - 0.1)
        self.display_pdf()

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFViewer(root)
    root.mainloop()
