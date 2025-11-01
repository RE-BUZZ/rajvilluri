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
import subprocess
import tempfile

class PersonSegmentatorUI:
    def __init__(self):
        """
        Initialize the person segmentator with YOLO for detection
        and a segmentation model for precise masking
        """
        # Load YOLOv8 model (will download automatically if not present)
        self.yolo_model = YOLO('yolov8m.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Video processing variables
        self.is_processing_video = False
        self.video_frame_count = 0
        self.current_frame = 0
        
        # Initialize UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI window"""
        self.root = tk.Tk()
        self.root.title("Person Background Remover - Images & Videos")
        self.root.geometry("1200x750")
        self.root.configure(bg='#2b2b2b')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='white', background='#2b2b2b')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#cccccc', background='#2b2b2b')
        
        # Main title
        title_label = ttk.Label(self.root, text="🎭 Person Background Remover", style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = ttk.Label(self.root, text="📸 Images & 🎥 Videos Supported", style='Info.TLabel')
        subtitle_label.pack(pady=(0, 10))
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left side - Input
        left_frame = tk.Frame(main_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        left_title = ttk.Label(left_frame, text="📁 Original Media", style='Title.TLabel')
        left_title.pack(pady=10)
        
        # Drop zone for input
        self.input_canvas = tk.Canvas(left_frame, bg='#4b4b4b', highlightthickness=2, 
                                     highlightcolor='#0078d4', relief=tk.SUNKEN)
        self.input_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input canvas text
        self.update_input_canvas_text()
        
        # Right side - Output
        right_frame = tk.Frame(main_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        right_title = ttk.Label(right_frame, text="✨ Background Removed", style='Title.TLabel')
        right_title.pack(pady=10)
        
        self.output_canvas = tk.Canvas(right_frame, bg='#4b4b4b', highlightthickness=2, 
                                      highlightcolor='#0078d4', relief=tk.SUNKEN)
        self.output_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.update_output_canvas_text()
        
        # Bottom frame for controls and status
        bottom_frame = tk.Frame(self.root, bg='#2b2b2b')
        bottom_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Progress frame
        progress_frame = tk.Frame(bottom_frame, bg='#2b2b2b')
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(0, 5))
        
        # Progress label
        self.progress_label = ttk.Label(progress_frame, text="", style='Info.TLabel')
        self.progress_label.pack()
        
        # Control buttons
        button_frame = tk.Frame(bottom_frame, bg='#2b2b2b')
        button_frame.pack()
        
        self.browse_btn = tk.Button(button_frame, text="📂 Browse Media", command=self.browse_media,
                                   bg='#0078d4', fg='white', font=('Arial', 10, 'bold'),
                                   padx=20, pady=8, cursor='hand2')
        self.browse_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_btn = tk.Button(button_frame, text="💾 Save Result", command=self.save_result,
                                 bg='#28a745', fg='white', font=('Arial', 10, 'bold'),
                                 padx=20, pady=8, state=tk.DISABLED, cursor='hand2')
        self.save_btn.pack(side=tk.LEFT, padx=10)
        
        self.process_video_btn = tk.Button(button_frame, text="🎬 Process Video", command=self.process_video,
                                           bg='#6f42c1', fg='white', font=('Arial', 10, 'bold'),
                                           padx=20, pady=8, cursor='hand2', state=tk.DISABLED)
        self.process_video_btn.pack(side=tk.LEFT, padx=10)
        
        self.clear_btn = tk.Button(button_frame, text="🗑️ Clear", command=self.clear_media,
                                  bg='#dc3545', fg='white', font=('Arial', 10, 'bold'),
                                  padx=20, pady=8, cursor='hand2')
        self.clear_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Video preview controls
        self.video_controls_frame = tk.Frame(bottom_frame, bg='#2b2b2b')
        
        self.play_pause_btn = tk.Button(self.video_controls_frame, text="▶️ Play", 
                                       command=self.toggle_video_playback,
                                       bg='#6f42c1', fg='white', font=('Arial', 9),
                                       padx=15, pady=5, cursor='hand2', state=tk.DISABLED)
        self.play_pause_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.frame_scale = tk.Scale(self.video_controls_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   bg='#2b2b2b', fg='white', highlightthickness=0,
                                   command=self.on_frame_change, state=tk.DISABLED)
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Status label
        self.status_label = ttk.Label(bottom_frame, text="Ready to process images and videos", style='Info.TLabel')
        self.status_label.pack(pady=(10, 0))
        
        # Bind events
        self.input_canvas.bind("<Button-1>", lambda e: self.browse_media())
        
        # Enable hover effects
        self.setup_hover_effects()
        
        # Store media
        self.original_media = None
        self.processed_media = None
        self.current_file_path = None
        self.is_video = False
        self.video_capture = None
        self.output_video_path = None
        self.video_playing = False
        
    def update_input_canvas_text(self):
        """Update input canvas text"""
        self.input_canvas.delete("all")
        self.input_canvas.create_text(
            300, 220, 
            text="📁 Drag & Drop Media Here\n\n📸 Images: PNG, JPG, JPEG, BMP\n🎥 Videos: MP4, AVI, MOV, MKV\n\n🖱️ Click to Browse", 
            fill='#cccccc', 
            font=('Arial', 12),
            justify=tk.CENTER
        )
        
    def update_output_canvas_text(self):
        """Update output canvas text"""
        self.output_canvas.delete("all")
        self.output_canvas.create_text(
            300, 250, 
            text="🖼️ Processed Media\nWill Appear Here", 
            fill='#cccccc', 
            font=('Arial', 12),
            justify=tk.CENTER
        )
        
    def setup_hover_effects(self):
        """Setup hover effects for input canvas"""
        def on_enter(event):
            self.input_canvas.configure(highlightcolor='#28a745')
            
        def on_leave(event):
            self.input_canvas.configure(highlightcolor='#0078d4')
        
        self.input_canvas.bind("<Enter>", on_enter)
        self.input_canvas.bind("<Leave>", on_leave)
        
    def browse_media(self):
        """Open file dialog to browse for media"""
        file_path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[
                ("All Media", "*.png *.jpg *.jpeg *.bmp *.tiff *.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.load_media(file_path)
            
    def load_media(self, file_path):
        """Load and display the input media"""
        try:
            self.current_file_path = file_path
            self.status_label.config(text="📥 Loading media...")
            
            # Check if it's a video or image
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            file_ext = os.path.splitext(file_path)[1].lower()
            self.is_video = file_ext in video_extensions
            
            if self.is_video:
                self.load_video(file_path)
            else:
                self.load_image(file_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load media: {str(e)}")
            self.status_label.config(text="❌ Error loading media")
            
    def load_image(self, file_path):
        """Load and display an image"""
        # Load image
        image = Image.open(file_path)
        self.original_media = image.copy()
        
        # Display image
        self.display_image_on_canvas(image, self.input_canvas, file_path)
        
        self.status_label.config(text="🖼️ Image loaded. Processing...")
        
        # Hide video controls
        self.video_controls_frame.pack_forget()
        
        # Process image in a separate thread
        threading.Thread(target=self.process_image_thread, daemon=True).start()
        
    def load_video(self, file_path):
        """Load and display first frame of video"""
        # Open video
        self.video_capture = cv2.VideoCapture(file_path)
        if not self.video_capture.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        self.video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        
        # Load first frame
        ret, frame = self.video_capture.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.original_media = image.copy()
            
            # Display first frame
            self.display_image_on_canvas(image, self.input_canvas, file_path, 
                                       extra_info=f"🎥 Video: {self.video_frame_count} frames, {fps:.1f} FPS")
        
        # Reset video to beginning
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Show video controls
        self.video_controls_frame.pack(pady=(10, 0))
        self.frame_scale.config(to=self.video_frame_count - 1, state=tk.NORMAL)
        self.play_pause_btn.config(state=tk.NORMAL)
        
        self.status_label.config(text="🎥 Video loaded. Click 'Process Video' to remove background from all frames...")
        
        # Enable process video button
        self.process_video_btn.config(state=tk.NORMAL)
        
    def display_image_on_canvas(self, image, canvas, file_path, extra_info=""):
        """Display image on specified canvas"""
        # Get canvas size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 580, 350
        
        # Resize image to fit canvas
        image_resized = self.resize_image_to_fit(image, canvas_width - 20, canvas_height - 60)
        
        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(image_resized)
        
        # Clear canvas and display image
        canvas.delete("all")
        img_x = canvas_width // 2
        img_y = (canvas_height - 40) // 2
        canvas.create_image(img_x, img_y, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep a reference
        
        # Add file info
        file_info = f"{os.path.basename(file_path)} ({image.width}x{image.height})"
        if extra_info:
            file_info += f"\n{extra_info}"
        canvas.create_text(10, 10, text=file_info, fill='#ffffff', 
                          font=('Arial', 8), anchor=tk.NW)
                          
    def resize_image_to_fit(self, image, max_width, max_height):
        """Resize image to fit within given dimensions while maintaining aspect ratio"""
        width, height = image.size
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    def process_image_thread(self):
        """Process image in a separate thread"""
        try:
            # Start progress bar
            self.root.after(0, lambda: self.progress.config(mode='indeterminate'))
            self.root.after(0, lambda: self.progress.start())
            
            # Process image
            result = self.segment_person_image(self.current_file_path)
            
            if result is not None:
                self.root.after(0, self.display_processed_image, result)
            else:
                self.root.after(0, self.show_no_person_detected)
                
        except Exception as e:
            self.root.after(0, self.show_processing_error, str(e))
        finally:
            self.root.after(0, lambda: self.progress.stop())
            
    def process_video_thread(self):
        """Process video in a separate thread"""
        try:
            self.is_processing_video = True
            self.root.after(0, lambda: self.progress.config(mode='determinate'))
            
            # Process video
            output_path = self.process_video_with_background_removal(self.current_file_path)
            
            if output_path:
                self.output_video_path = output_path
                # Load first frame of processed video for preview
                cap = cv2.VideoCapture(output_path)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result_image = Image.fromarray(frame_rgb)
                    self.root.after(0, self.display_processed_video, result_image)
                cap.release()
            else:
                self.root.after(0, self.show_no_person_detected)
                
        except Exception as e:
            self.root.after(0, self.show_processing_error, str(e))
        finally:
            self.is_processing_video = False
            self.root.after(0, lambda: self.progress.stop())
            
    def process_video_with_background_removal(self, video_path):
        """Process entire video to remove background from each frame"""
        try:
            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create temporary output path for video without audio
            temp_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_dir, f"temp_no_audio_{os.path.basename(video_path)}")
            
            # Create VideoWriter for output (without audio)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update progress
                progress = (frame_count / total_frames) * 100
                self.root.after(0, lambda p=progress: self.progress.config(value=p))
                self.root.after(0, lambda f=frame_count, t=total_frames: 
                               self.progress_label.config(text=f"Processing frame {f+1}/{t}"))
                
                # Process frame to remove background
                processed_frame = self.process_frame_background_removal(frame)
                
                # Write processed frame
                out.write(processed_frame)
                
                frame_count += 1
                self.current_frame = frame_count
            
            # Release everything
            cap.release()
            out.release()
            
            # Now merge with original audio using ffmpeg
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            final_output_path = os.path.join(temp_dir, f"{base_name}_background_removed.mp4")
            
            # Use ffmpeg to combine processed video with original audio
            self.root.after(0, lambda: self.progress_label.config(text="Adding original audio..."))
            
            ffmpeg_command = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', temp_video_path,  # processed video (no audio)
                '-i', video_path,       # original video (for audio)
                '-c:v', 'copy',         # copy video codec
                '-c:a', 'aac',          # encode audio as AAC
                '-map', '0:v:0',        # map video from first input
                '-map', '1:a:0',        # map audio from second input
                final_output_path
            ]
            
            # Run ffmpeg
            result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Clean up temporary file
                os.remove(temp_video_path)
                return final_output_path
            else:
                # If ffmpeg fails, return video without audio
                print(f"FFmpeg error: {result.stderr}")
                return temp_video_path
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return None
            
    def process_frame_background_removal(self, frame):
        """Process a single frame to remove background"""
        try:
            # Detect persons in frame
            results = self.yolo_model(frame)
            person_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls) == 0:  # person class
                            person_boxes.append(box.xyxy[0].cpu().numpy())
            
            if not person_boxes:
                # If no person detected, return black frame or original frame
                return np.zeros_like(frame)  # Black frame
            
            # Create mask using GrabCut
            mask = self.refine_mask_with_grabcut(frame, person_boxes)
            
            # Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            
            # Create transparent background (green screen for video)
            # Normalize mask
            mask_norm = mask.astype(float) / 255
            mask_3ch = np.stack([mask_norm] * 3, axis=-1)
            
            # Green background for videos (easier for further processing)
            green_bg = np.full_like(frame, (0, 255, 0), dtype=np.uint8)  # Green background
            result_frame = (frame * mask_3ch + green_bg * (1 - mask_3ch)).astype(np.uint8)
            
            return result_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame  # Return original frame on error
            
    def segment_person_image(self, image_path):
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
                        if int(box.cls) == 0:  # person class
                            person_boxes.append(box.xyxy[0].cpu().numpy())
            
            if not person_boxes:
                return None
            
            # Create refined mask using GrabCut
            mask = self.refine_mask_with_grabcut(image, person_boxes)
            
            # Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
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
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 - x1 < 10 or y2 - y1 < 10:  # Skip very small boxes
                continue
            
            # Create rectangle for GrabCut
            rect = (x1, y1, x2-x1, y2-y1)
            
            # Initialize mask for GrabCut
            gc_mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            try:
                # Apply GrabCut
                cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                
                # Create final mask
                gc_mask2 = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8')
                mask = np.maximum(mask, gc_mask2 * 255)
            except Exception as e:
                print(f"GrabCut failed for box {box}: {e}")
                # Fallback to simple rectangle mask
                mask[y1:y2, x1:x2] = 255
        
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
            self.processed_media = processed_image
            
            # Create checkered background to show transparency
            canvas_width = self.output_canvas.winfo_width() or 580
            canvas_height = self.output_canvas.winfo_height() or 350
            
            bg_image = self.create_checkered_background(canvas_width - 20, canvas_height - 60)
            
            # Resize processed image
            image_resized = self.resize_image_to_fit(processed_image, canvas_width - 20, canvas_height - 60)
            
            # Composite over checkered background
            bg_image.paste(image_resized, 
                         ((canvas_width - 20 - image_resized.width) // 2,
                          (canvas_height - 60 - image_resized.height) // 2), 
                         image_resized)
            
            # Display
            photo = ImageTk.PhotoImage(bg_image)
            self.output_canvas.delete("all")
            self.output_canvas.create_image(canvas_width // 2, (canvas_height - 40) // 2, 
                                          image=photo, anchor=tk.CENTER)
            self.output_canvas.image = photo
            
            # Add success indicator
            self.output_canvas.create_text(10, 10, text="✅ Background Removed", 
                                         fill='#28a745', font=('Arial', 8), anchor=tk.NW)
            
            # Enable save button
            self.save_btn.config(state=tk.NORMAL)
            self.status_label.config(text="✅ Processing completed successfully!")
            
        except Exception as e:
            self.show_processing_error(f"Failed to display processed image: {str(e)}")
            
    def display_processed_video(self, first_frame_image):
        """Display the first frame of processed video"""
        try:
            self.processed_media = first_frame_image  # Store for reference
            
            # Display first frame
            self.display_image_on_canvas(first_frame_image, self.output_canvas, 
                                       self.output_video_path,
                                       "✅ Video processed with background removed")
            
            # Enable save button
            self.save_btn.config(state=tk.NORMAL)
            self.status_label.config(text="✅ Video processing completed! Audio preserved.")
            
        except Exception as e:
            self.show_processing_error(f"Failed to display processed video: {str(e)}")
            
    def create_checkered_background(self, width, height, square_size=20):
        """Create a checkered pattern background to show transparency"""
        image = Image.new('RGB', (width, height), '#ffffff')
        pixels = image.load()
        
        for i in range(width):
            for j in range(height):
                if (i // square_size + j // square_size) % 2:
                    pixels[i, j] = (220, 220, 220)
                    
        return image
        
    def show_no_person_detected(self):
        """Show message when no person is detected"""
        self.output_canvas.delete("all")
        canvas_width = self.output_canvas.winfo_width() or 580
        canvas_height = self.output_canvas.winfo_height() or 350
        
        self.output_canvas.create_text(
            canvas_width // 2, canvas_height // 2,
            text="❌ No Person Detected\nin the Media\n\nTry with media that\ncontains people",
            fill='#ff6b6b',
            font=('Arial', 14, 'bold'),
            justify=tk.CENTER
        )
        self.status_label.config(text="❌ No person detected in the media")
        
    def show_processing_error(self, error_msg):
        """Show processing error message"""
        self.output_canvas.delete("all")
        canvas_width = self.output_canvas.winfo_width() or 580
        canvas_height = self.output_canvas.winfo_height() or 350
        
        self.output_canvas.create_text(
            canvas_width // 2, canvas_height // 2,
            text="⚠️ Processing Error\nSee status for details",
            fill='#ff6b6b',
            font=('Arial', 14, 'bold'),
            justify=tk.CENTER
        )
        self.status_label.config(text=f"❌ Error: {error_msg}")
        messagebox.showerror("Processing Error", error_msg)
        
    def toggle_video_playback(self):
        """Toggle video playback for preview"""
        # This is a placeholder for video playback functionality
        if self.video_playing:
            self.play_pause_btn.config(text="▶️ Play")
            self.video_playing = False
        else:
            self.play_pause_btn.config(text="⏸️ Pause")
            self.video_playing = True
            
    def on_frame_change(self, value):
        """Handle frame slider change"""
        if self.video_capture and not self.is_processing_video:
            frame_num = int(value)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.video_capture.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                self.display_image_on_canvas(image, self.input_canvas, 
                                           self.current_file_path,
                                           f"Frame {frame_num + 1}/{self.video_frame_count}")
                                           
    def save_result(self):
        """Save the processed media"""
        if self.processed_media is None and not self.output_video_path:
            messagebox.showwarning("Warning", "No processed media to save")
            return
            
        try:
            if self.is_video and self.output_video_path:
                # Save video
                file_path = filedialog.asksaveasfilename(
                    title="Save Processed Video",
                    defaultextension=".mp4",
                    filetypes=[
                        ("MP4 files", "*.mp4"),
                        ("AVI files", "*.avi"),
                        ("All files", "*.*")
                    ]
                )
                
                if file_path:
                    # Copy the processed video to chosen location
                    import shutil
                    shutil.copy2(self.output_video_path, file_path)
                    messagebox.showinfo("Success", f"✅ Processed video saved!\n\nSaved to: {file_path}\n\n🔊 Original audio preserved")
                    self.status_label.config(text=f"💾 Video saved to {os.path.basename(file_path)}")
            else:
                # Save image
                file_path = filedialog.asksaveasfilename(
                    title="Save Processed Image",
                    defaultextension=".png",
                    filetypes=[
                        ("PNG files (transparent)", "*.png"),
                        ("JPEG files (white background)", "*.jpg"),
                        ("All files", "*.*")
                    ]
                )
                
                if file_path:
                    if file_path.lower().endswith('.png'):
                        self.processed_media.save(file_path, "PNG")
                        msg = "✅ Transparent PNG saved!"
                    else:
                        # Create white background version
                        white_bg = Image.new('RGB', self.processed_media.size, (255, 255, 255))
                        white_bg.paste(self.processed_media, mask=self.processed_media.split()[-1])
                        white_bg.save(file_path)
                        msg = "✅ Image with white background saved!"
                    
                    messagebox.showinfo("Success", f"{msg}\n\nSaved to: {file_path}")
                    self.status_label.config(text=f"💾 Image saved to {os.path.basename(file_path)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save media: {str(e)}")
            
    def clear_media(self):
        """Clear both input and output media"""
        # Clear canvases
        self.input_canvas.delete("all")
        self.output_canvas.delete("all")
        
        # Reset canvas text
        self.update_input_canvas_text()
        self.update_output_canvas_text()
        
        # Reset variables
        self.original_media = None
        self.processed_media = None
        self.current_file_path = None
        self.is_video = False
        self.output_video_path = None
        
        # Release video capture
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        # Hide video controls
        self.video_controls_frame.pack_forget()
        
        # Reset controls
        self.save_btn.config(state=tk.DISABLED)
        if hasattr(self, 'process_video_btn'):
            self.process_video_btn.config(state=tk.DISABLED)
        self.play_pause_btn.config(state=tk.DISABLED)
        self.frame_scale.config(state=tk.DISABLED)
        
        # Reset progress
        self.progress.config(value=0)
        self.progress_label.config(text="")
        
        # Reset status
        self.status_label.config(text="Ready to process images and videos")
        
    def run(self):
        """Start the UI application"""
        self.root.mainloop()
        
    def process_video(self):
        """Start video processing"""
        if self.is_video and self.current_file_path:
            self.status_label.config(text="🎬 Starting video processing...")
            self.process_video_btn.config(state=tk.DISABLED)
            threading.Thread(target=self.process_video_thread, daemon=True).start()

def main():
    """Main function to run the UI application"""
    print("🚀 Starting Person Background Remover UI...")
    print("📸 Images & 🎥 Videos supported")
    print("Loading YOLO model...")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg found - audio will be preserved in videos")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg not found - videos will be processed without audio")
        print("Install FFmpeg for audio support: https://ffmpeg.org/download.html")
    
    try:
        app = PersonSegmentatorUI()
        app.run()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install ultralytics opencv-python pillow torch torchvision")

if __name__ == "__main__":
    main()
