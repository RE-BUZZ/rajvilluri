import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import Canvas
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import threading

class PersonSegmentatorUI:
    def __init__(self):
        """
        Initialize the person segmentator with YOLO for detection
        and a segmentation model for precise masking
        """
        # Load YOLOv8 model (will download automatically if not present)
        self.yolo_model = YOLO('yolov8n.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI window"""
        self.root = tk.Tk()
        self.root.title("Person Background Remover")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2b2b2b')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='white', background='#2b2b2b')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#cccccc', background='#2b2b2b')
        
        # Main title
        title_label = ttk.Label(self.root, text="Person Background Remover", style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left side - Input image
        left_frame = tk.Frame(main_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        left_title = ttk.Label(left_frame, text="Original Image", style='Title.TLabel')
        left_title.pack(pady=10)
        
        # Drop zone for input image
        self.input_canvas = tk.Canvas(left_frame, bg='#4b4b4b', highlightthickness=2, 
                                     highlightcolor='#0078d4', relief=tk.SUNKEN)
        self.input_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Drag and drop text
        self.input_canvas.create_text(
            300, 250, 
            text="Drag & Drop Image Here\nor\nClick to Browse", 
            fill='#cccccc', 
            font=('Arial', 14),
            justify=tk.CENTER
        )
        
        # Right side - Output image
        right_frame = tk.Frame(main_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        right_title = ttk.Label(right_frame, text="Background Removed", style='Title.TLabel')
        right_title.pack(pady=10)
        
        self.output_canvas = tk.Canvas(right_frame, bg='#4b4b4b', highlightthickness=2, 
                                      highlightcolor='#0078d4', relief=tk.SUNKEN)
        self.output_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.output_canvas.create_text(
            300, 250, 
            text="Processed Image\nWill Appear Here", 
            fill='#cccccc', 
            font=('Arial', 14),
            justify=tk.CENTER
        )
        
        # Bottom frame for controls and status
        bottom_frame = tk.Frame(self.root, bg='#2b2b2b')
        bottom_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(bottom_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Control buttons
        button_frame = tk.Frame(bottom_frame, bg='#2b2b2b')
        button_frame.pack()
        
        self.browse_btn = tk.Button(button_frame, text="Browse Image", command=self.browse_image,
                                   bg='#0078d4', fg='white', font=('Arial', 10, 'bold'),
                                   padx=20, pady=5)
        self.browse_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_btn = tk.Button(button_frame, text="Save Result", command=self.save_result,
                                 bg='#28a745', fg='white', font=('Arial', 10, 'bold'),
                                 padx=20, pady=5, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=10)
        
        self.clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_images,
                                  bg='#dc3545', fg='white', font=('Arial', 10, 'bold'),
                                  padx=20, pady=5)
        self.clear_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Status label
        self.status_label = ttk.Label(bottom_frame, text="Ready to process images", style='Info.TLabel')
        self.status_label.pack(pady=(10, 0))
        
        # Bind events
        self.input_canvas.bind("<Button-1>", lambda e: self.browse_image())
        
        # Enable drag and drop
        self.setup_drag_drop()
        
        # Store images
        self.original_image = None
        self.processed_image = None
        self.current_file_path = None
        
    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        def on_drag_enter(event):
            self.input_canvas.configure(highlightcolor='#28a745')
            
        def on_drag_leave(event):
            self.input_canvas.configure(highlightcolor='#0078d4')
            
        def on_drop(event):
            # This is a simplified drag-drop handler
            # In a real implementation, you'd need tkinterdnd2 package
            pass
            
        # Bind drag events
        self.input_canvas.bind("<Enter>", on_drag_enter)
        self.input_canvas.bind("<Leave>", on_drag_leave)
        
    def browse_image(self):
        """Open file dialog to browse for image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.load_image(file_path)
            
    def load_image(self, file_path):
        """Load and display the input image"""
        try:
            self.current_file_path = file_path
            self.status_label.config(text="Loading image...")
            
            # Load image
            image = Image.open(file_path)
            self.original_image = image.copy()
            
            # Resize image to fit canvas while maintaining aspect ratio
            canvas_width = self.input_canvas.winfo_width()
            canvas_height = self.input_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet drawn, use default size
                canvas_width, canvas_height = 400, 300
            
            image_resized = self.resize_image_to_fit(image, canvas_width - 20, canvas_height - 20)
            
            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(image_resized)
            
            # Clear canvas and display image
            self.input_canvas.delete("all")
            img_x = canvas_width // 2
            img_y = canvas_height // 2
            self.input_canvas.create_image(img_x, img_y, image=photo, anchor=tk.CENTER)
            self.input_canvas.image = photo  # Keep a reference
            
            self.status_label.config(text="Image loaded. Processing...")
            
            # Process image in a separate thread
            threading.Thread(target=self.process_image_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_label.config(text="Error loading image")
            
    def resize_image_to_fit(self, image, max_width, max_height):
        """Resize image to fit within given dimensions while maintaining aspect ratio"""
        width, height = image.size
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    def process_image_thread(self):
        """Process the image in a separate thread"""
        try:
            # Start progress bar
            self.root.after(0, lambda: self.progress.start())
            
            # Process image using the segmentation logic
            result = self.segment_person(self.current_file_path)
            
            if result is not None:
                # Update UI in main thread
                self.root.after(0, self.display_processed_image, result)
            else:
                self.root.after(0, self.show_no_person_detected)
                
        except Exception as e:
            self.root.after(0, self.show_processing_error, str(e))
        finally:
            self.root.after(0, lambda: self.progress.stop())
            
    def segment_person(self, image_path):
        """Segment person from image and return processed image"""
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Detect persons
            results = self.yolo_model(image)
            person_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Class 0 is 'person' in COCO dataset
                        if int(box.cls) == 0:  # person class
                            person_boxes.append(box.xyxy[0].cpu().numpy())
            
            if not person_boxes:
                return None
            
            # Create refined mask using GrabCut
            mask = self.refine_mask_with_grabcut(image, person_boxes)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Apply Gaussian blur to soften edges
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            
            # Create transparent image
            return self.create_transparent_image(image, mask)
            
        except Exception as e:
            print(f"Error in segmentation: {e}")
            return None
            
    def refine_mask_with_grabcut(self, image, person_boxes):
        """Use GrabCut algorithm to refine the person segmentation"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for box in person_boxes:
            x1, y1, x2, y2 = box.astype(int)
            
            # Create rectangle for GrabCut
            rect = (x1, y1, x2-x1, y2-y1)
            
            # Initialize mask for GrabCut
            gc_mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create final mask
            gc_mask2 = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8')
            mask = np.maximum(mask, gc_mask2 * 255)
        
        return mask
        
    def create_transparent_image(self, image, mask):
        """Create transparent PNG image"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create 4-channel image (RGBA)
        h, w = image.shape[:2]
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[:, :, :3] = image_rgb
        result[:, :, 3] = mask  # Alpha channel
        
        # Convert to PIL Image
        return Image.fromarray(result, 'RGBA')
        
    def display_processed_image(self, processed_image):
        """Display the processed image on the right canvas"""
        try:
            self.processed_image = processed_image
            
            # Resize image to fit canvas
            canvas_width = self.output_canvas.winfo_width()
            canvas_height = self.output_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 400, 300
            
            image_resized = self.resize_image_to_fit(processed_image, canvas_width - 20, canvas_height - 20)
            
            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(image_resized)
            
            # Clear canvas and display image
            self.output_canvas.delete("all")
            img_x = canvas_width // 2
            img_y = canvas_height // 2
            self.output_canvas.create_image(img_x, img_y, image=photo, anchor=tk.CENTER)
            self.output_canvas.image = photo  # Keep a reference
            
            # Enable save button
            self.save_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Processing completed successfully!")
            
        except Exception as e:
            self.show_processing_error(f"Failed to display processed image: {str(e)}")
            
    def show_no_person_detected(self):
        """Show message when no person is detected"""
        self.output_canvas.delete("all")
        canvas_width = self.output_canvas.winfo_width() or 400
        canvas_height = self.output_canvas.winfo_height() or 300
        
        self.output_canvas.create_text(
            canvas_width // 2, canvas_height // 2,
            text="No Person Detected\nin the Image",
            fill='#ff6b6b',
            font=('Arial', 14, 'bold'),
            justify=tk.CENTER
        )
        self.status_label.config(text="No person detected in the image")
        
    def show_processing_error(self, error_msg):
        """Show processing error message"""
        self.output_canvas.delete("all")
        canvas_width = self.output_canvas.winfo_width() or 400
        canvas_height = self.output_canvas.winfo_height() or 300
        
        self.output_canvas.create_text(
            canvas_width // 2, canvas_height // 2,
            text="Processing Error\nSee status for details",
            fill='#ff6b6b',
            font=('Arial', 14, 'bold'),
            justify=tk.CENTER
        )
        self.status_label.config(text=f"Error: {error_msg}")
        messagebox.showerror("Processing Error", error_msg)
        
    def save_result(self):
        """Save the processed image"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Processed Image",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Save with transparent background for PNG, white background for other formats
                if file_path.lower().endswith('.png'):
                    self.processed_image.save(file_path, "PNG")
                else:
                    # Create white background version
                    white_bg = Image.new('RGB', self.processed_image.size, (255, 255, 255))
                    white_bg.paste(self.processed_image, mask=self.processed_image.split()[-1])
                    white_bg.save(file_path)
                
                messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
                self.status_label.config(text=f"Image saved to {os.path.basename(file_path)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
            
    def clear_images(self):
        """Clear both input and output images"""
        # Clear canvases
        self.input_canvas.delete("all")
        self.output_canvas.delete("all")
        
        # Reset canvas text
        self.input_canvas.create_text(
            300, 250, 
            text="Drag & Drop Image Here\nor\nClick to Browse", 
            fill='#cccccc', 
            font=('Arial', 14),
            justify=tk.CENTER
        )
        
        self.output_canvas.create_text(
            300, 250, 
            text="Processed Image\nWill Appear Here", 
            fill='#cccccc', 
            font=('Arial', 14),
            justify=tk.CENTER
        )
        
        # Reset variables
        self.original_image = None
        self.processed_image = None
        self.current_file_path = None
        
        # Disable save button
        self.save_btn.config(state=tk.DISABLED)
        
        # Reset status
        self.status_label.config(text="Ready to process images")
        
    def run(self):
        """Start the UI application"""
        self.root.mainloop()

def main():
    """Main function to run the UI application"""
    print("Starting Person Background Remover UI...")
    print("Loading YOLO model...")
    
    try:
        app = PersonSegmentatorUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install ultralytics opencv-python pillow torch torchvision")

if __name__ == "__main__":
    main()
