import numpy as np
import tkinter as tk
from tkinter import simpledialog, ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import matplotlib.pyplot as plt # Still used for the figure, but not for weight viz
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import pickle # For records and labels if not saving everything with torch.save

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

# --- Constants ---
MODEL_FILENAME = "drawing_cnn_model.pth"
APP_VERSION = "2.0.0" # For model compatibility

# --- Convolutional Neural Network (CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=28):
        super(SimpleCNN, self).__init__()
        self.input_size = input_size
        # Calculate expected size after convolutions and pooling
        # For 28x28 input:
        # Conv1 (28x28 -> 26x26 if kernel 3, no pad -> 28x28 if kernel 3, pad 1)
        # Pool1 (28x28 -> 14x14 if kernel 2, stride 2)
        # Conv2 (14x14 -> 12x12 if kernel 3, no pad -> 14x14 if kernel 3, pad 1)
        # Pool2 (14x14 -> 7x7 if kernel 2, stride 2)
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # Output: (N, 16, 28, 28)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: (N, 16, 14, 14)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Output: (N, 32, 14, 14)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: (N, 32, 7, 7)
        
        # Calculate the flattened size
        self.flattened_size = 32 * (input_size // 4) * (input_size // 4) # after two 2x2 pools
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Logits
        return x

    def update_output_layer(self, new_num_classes, device='cpu'):
        old_out_features = self.fc2.out_features
        old_weights = self.fc2.weight.data
        old_biases = self.fc2.bias.data

        new_fc2 = nn.Linear(self.fc1.out_features, new_num_classes).to(device)

        # Copy old weights and biases
        min_features = min(old_out_features, new_num_classes)
        new_fc2.weight.data[:min_features, :] = old_weights[:min_features, :]
        new_fc2.bias.data[:min_features] = old_biases[:min_features]
        
        self.fc2 = new_fc2
        self.num_classes = new_num_classes
        print(f"CNN output layer updated to {new_num_classes} classes.")


# --- Main Application ---
class DrawApp:
    def __init__(self, master, initial_labels, required=10):
        self.master = master
        self.labels = initial_labels[:]
        self.required = required
        self.size, self.cs = 28, 280 # NN input size (28x28), Canvas size
        self.bg, self.fg = 'white','black'
        self.records = []
        self.last_conf = 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize NN (num_classes will be updated in load_data)
        self.nn = SimpleCNN(num_classes=len(self.labels) if self.labels else 1).to(self.device)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=0.001) # Default LR for Adam
        self.criterion = nn.CrossEntropyLoss()

        # Image preprocessing for CNN
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(), # Converts to [C, H, W] and scales to [0,1]
            # For MNIST-like data (black lines on white bg initially):
            # ToTensor converts PIL (0-255) to (0-1). If drawing is black (0) on white (255),
            # and NN expects lines to be high values, an inversion is needed.
            # My get_nn_input will handle inversion before ToTensor.
            transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1,1] (common for CNNs)
        ])


        # --- Notebook and Frames ---
        self.nb = ttk.Notebook(master)
        self.draw_f = tk.Frame(self.nb)
        self.stats_f = tk.Frame(self.nb)
        self.nb.add(self.draw_f, text="Draw")
        self.nb.add(self.stats_f, text="Statistics")
        self.nb.pack(fill='both', expand=True)

        # --- Draw Tab ---
        self.canvas = tk.Canvas(self.draw_f, width=self.cs, height=self.cs, bg=self.bg)
        self.canvas.grid(row=0, columnspan=7)
        self.pil_image_draw = Image.new('L', (self.cs, self.cs), 255) # For drawing
        self.draw_tool = ImageDraw.Draw(self.pil_image_draw)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

        cmds = [
            ("Clear", self.clear), ("Capture", self.capture_sample),
            ("Predict", self.predict), ("Confirm", self.confirm),
            ("Teach", self.teach), ("Load Image", self.load_image_file),
            ("Quit", self.on_quit),
        ]
        self.btns = {}
        for i,(txt,cmd) in enumerate(cmds):
            b = tk.Button(self.draw_f, text=txt, width=10, command=cmd)
            b.grid(row=1, column=i, padx=2, pady=4)
            self.btns[txt] = b

        self.label_var = tk.StringVar(value="— Initializing... —")
        tk.Label(self.draw_f, textvariable=self.label_var, font=('Arial',12)).grid(row=2, columnspan=7, pady=4)

        # --- Stats Tab ---
        # Text stats on the left
        self.stats_text_frame = tk.Frame(self.stats_f)
        self.stats_text_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        tk.Button(self.stats_text_frame, text="Refresh Stats", command=self.show_stats).pack(pady=4, anchor='n')
        self.stats_text_widget = tk.Text(self.stats_text_frame, width=40, height=20, wrap=tk.WORD)
        self.stats_text_widget.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        # Scrollable thumbnail gallery on the right
        self.gallery_frame_container = tk.Frame(self.stats_f)
        self.gallery_frame_container.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        self.thumb_canvas = tk.Canvas(self.gallery_frame_container, borderwidth=0)
        self.thumb_scrollbar = tk.Scrollbar(self.gallery_frame_container, orient="vertical", command=self.thumb_canvas.yview)
        self.scrollable_thumb_frame = tk.Frame(self.thumb_canvas) # Frame to hold thumbnails

        self.scrollable_thumb_frame.bind(
            "<Configure>",
            lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all"))
        )
        self.thumb_canvas_window = self.thumb_canvas.create_window((0, 0), window=self.scrollable_thumb_frame, anchor="nw")
        self.thumb_canvas.configure(yscrollcommand=self.thumb_scrollbar.set)

        self.thumb_canvas.pack(side="left", fill="both", expand=True)
        self.thumb_scrollbar.pack(side="right", fill="y")
        
        self.thumb_canvas.bind('<Enter>', lambda e: self._bind_mousewheel(e, self.thumb_canvas))
        self.thumb_canvas.bind('<Leave>', lambda e: self._unbind_mousewheel(e, self.thumb_canvas))


        # --- Load data or Begin Initial Sampling ---
        self.load_data_and_initialize()
        self.master.protocol("WM_DELETE_WINDOW", self.on_quit)

    def _bind_mousewheel(self, event, widget_to_scroll):
        # For Linux and Windows
        widget_to_scroll.bind_all("<MouseWheel>", lambda e: self._on_mousewheel(e, widget_to_scroll)) # Windows
        widget_to_scroll.bind_all("<Button-4>", lambda e: self._on_mousewheel(e, widget_to_scroll)) # Linux scroll up
        widget_to_scroll.bind_all("<Button-5>", lambda e: self._on_mousewheel(e, widget_to_scroll)) # Linux scroll down

    def _unbind_mousewheel(self, event, widget_to_scroll):
        widget_to_scroll.unbind_all("<MouseWheel>")
        widget_to_scroll.unbind_all("<Button-4>")
        widget_to_scroll.unbind_all("<Button-5>")

    def _on_mousewheel(self, event, widget_to_scroll):
        if event.num == 4 or event.delta > 0: # Scroll up
            widget_to_scroll.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0: # Scroll down
            widget_to_scroll.yview_scroll(1, "units")

    def load_data_and_initialize(self):
            if os.path.exists(MODEL_FILENAME):
                try:
                    # MODIFICATION HERE: Added weights_only=False
                    checkpoint = torch.load(MODEL_FILENAME, map_location=self.device, weights_only=False)
                    
                    if checkpoint.get('app_version') != APP_VERSION:
                        messagebox.showwarning("Load Warning", f"Model file is from a different app version (v{checkpoint.get('app_version')}). Starting fresh to avoid compatibility issues.")
                        # Optionally, you might want to attempt to delete or backup the old file
                        # For now, we'll just proceed to fresh start after the warning if version mismatches.
                        raise FileNotFoundError(f"Version mismatch: expected {APP_VERSION}, got {checkpoint.get('app_version')}")

                    self.labels = checkpoint['labels']
                    self.records = checkpoint['records']
                    
                    self.nn = SimpleCNN(num_classes=len(self.labels)).to(self.device)
                    self.nn.load_state_dict(checkpoint['model_state_dict'])
                    
                    self.optimizer = optim.Adam(self.nn.parameters(), lr=0.001) 
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                    self.label_var.set("✅ Model loaded. You may now Predict.")
                    self._set_initial_mode(False)
                    self.show_stats()
                    print(f"Model loaded successfully from {MODEL_FILENAME}")
                    return
                except FileNotFoundError as e: # Catch version mismatch specifically
                    messagebox.showerror("Load Error", f"{str(e)}\nStarting fresh.")
                except Exception as e: # Catch other load errors (like the one in the image if weights_only isn't set)
                    messagebox.showerror("Load Error", f"Could not load model: {e}\nThis can happen if the model file is corrupted or incompatible. Starting fresh.")
            
            # --- Fresh Start ---
            # (The rest of the fresh start logic remains the same)
            if not self.labels:
                new_label = simpledialog.askstring("Initial Label", "No labels defined. Enter the first label name:", parent=self.master)
                if new_label and new_label.strip():
                    self.labels.append(new_label.strip())
                else:
                    messagebox.showinfo("Info", "Application needs at least one label. Exiting.")
                    self.master.quit()
                    return
            
            self.nn = SimpleCNN(num_classes=len(self.labels)).to(self.device)
            self.optimizer = optim.Adam(self.nn.parameters(), lr=0.001)
            self._init_label_idx = 0
            self._init_count = 0
            self._set_initial_mode(True)
            self._collect_initial_samples()

    def save_data(self):
        if not self.nn: return
        try:
            checkpoint = {
                'app_version': APP_VERSION,
                'labels': self.labels,
                'records': self.records,
                'model_state_dict': self.nn.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, MODEL_FILENAME)
            print(f"Model saved to {MODEL_FILENAME}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save model: {e}")

    def on_quit(self):
        self.save_data()
        self.master.quit()

    def _set_initial_mode(self, initial):
        if initial:
            if self._init_label_idx < len(self.labels):
                 self.label_var.set(f"Draw '{self.labels[self._init_label_idx]}' (Sample 1/{self.required})")
            else: self.label_var.set("Preparing initial samples...")
            for name, btn in self.btns.items():
                btn.config(state='normal' if name in ('Capture','Clear','Quit') else 'disabled')
        else:
            self.label_var.set("Ready. Draw and Predict, or Load Image.")
            for name, btn in self.btns.items():
                if name in ('Confirm', 'Teach'): btn.config(state='disabled')
                elif name == 'Capture': btn.config(state='disabled')
                else: btn.config(state='normal')

    def _collect_initial_samples(self):
        if self._init_label_idx >= len(self.labels):
            self.label_var.set("✅ Initial samples collected! Training model...")
            self.master.update_idletasks()
            
            if not self.records:
                self.label_var.set("No samples for training.")
                self._set_initial_mode(False)
                return

            # Prepare data for batch training
            inputs = []
            targets = []
            for record in self.records:
                # 'x_tensor' should be stored in records during capture_sample
                if 'x_tensor' in record and 'y_idx' in record:
                    inputs.append(record['x_tensor'])
                    targets.append(record['y_idx'])
            
            if not inputs:
                self.label_var.set("No valid tensor data for training.")
                self._set_initial_mode(False)
                return

            inputs_tensor = torch.stack(inputs).to(self.device) # Batch of tensors
            targets_tensor = torch.LongTensor(targets).to(self.device)

            # Batch Training Loop
            self.nn.train() # Set model to training mode
            epochs = 20 # Fewer epochs for initial quick training, can be increased
            batch_size = 10 
            num_batches = len(inputs_tensor) // batch_size
            
            print(f"Starting initial CNN training: {len(inputs_tensor)} samples, {epochs} epochs, batch size {batch_size}, LR {self.optimizer.param_groups[0]['lr']:.4f}")

            for epoch in range(epochs):
                epoch_loss = 0.0
                # Simple shuffling for demonstration, proper DataLoader is better for large datasets
                permutation = torch.randperm(inputs_tensor.size(0))
                
                for i in range(0, inputs_tensor.size(0), batch_size):
                    self.optimizer.zero_grad()
                    indices = permutation[i : i + batch_size]
                    batch_inputs, batch_targets = inputs_tensor[indices], targets_tensor[indices]
                    
                    outputs = self.nn(batch_inputs)
                    loss = self.criterion(outputs, batch_targets)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                
                avg_epoch_loss = epoch_loss / (num_batches if num_batches > 0 else 1)
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
                self.label_var.set(f"Training... Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.3f}")
                self.master.update_idletasks()


            self.label_var.set("✅ Initial training complete! You may now Predict.")
            self._set_initial_mode(False)
            self.show_stats()
            return

        current_label = self.labels[self._init_label_idx]
        sample_num = self._init_count + 1
        self.label_var.set(f"Draw '{current_label}' — sample {sample_num}/{self.required}, then Capture.")

    def paint(self, e):
        brush_size = 10
        x1,y1 = e.x - brush_size, e.y - brush_size
        x2,y2 = e.x + brush_size, e.y + brush_size
        self.canvas.create_oval(x1,y1,x2,y2, fill=self.fg, outline=self.fg)
        self.draw_tool.ellipse([x1,y1,x2,y2], fill=0) # Black on white for PIL

    def clear(self):
        self.canvas.delete('all')
        self.pil_image_draw = Image.new('L', (self.cs, self.cs), 255)
        self.draw_tool = ImageDraw.Draw(self.pil_image_draw)

    def get_nn_input_tensor(self, pil_image):
        # Preprocess the PIL image for the CNN
        # Invert colors: drawing is black (0) on white (255).
        # CNN might expect features (lines) to be high values.
        # ToTensor converts PIL (0-255) to (0-1).
        # If we invert first, lines become 255 (white), bg 0 (black).
        # Then ToTensor maps lines to 1, bg to 0.
        # Then Normalize maps lines to 1, bg to -1 if range is [-1,1].
        inverted_image = ImageOps.invert(pil_image.convert('L'))
        return self.transform(inverted_image).unsqueeze(0) # Add batch dimension

    def capture_sample(self):
        if self._init_label_idx >= len(self.labels): return

        # Get the current drawing as a PIL image (it's already self.pil_image_draw)
        current_drawing_pil = self.pil_image_draw.copy()
        nn_input_tensor = self.get_nn_input_tensor(current_drawing_pil)
        
        true_label_name = self.labels[self._init_label_idx]
        true_label_idx = self._init_label_idx # y_idx for CrossEntropyLoss

        self.records.append({
            'pil_image': current_drawing_pil.resize((50,50), Image.Resampling.LANCZOS), # Store resized for gallery
            'x_tensor': nn_input_tensor.squeeze(0).cpu(), # Store tensor without batch dim, on CPU
            'y_idx': true_label_idx, # Store target index for training
            'true_label_name': true_label_name,
            'pred_label_name': None, 'correct': None, 'conf': None
        })
        self._init_count += 1
        self.clear()
        
        if self._init_count >= self.required:
            self._init_count = 0
            self._init_label_idx += 1
        self._collect_initial_samples()

    def predict(self):
        self.nn.eval() # Set model to evaluation mode
        with torch.no_grad(): # No need to track gradients for prediction
            input_tensor = self.get_nn_input_tensor(self.pil_image_draw).to(self.device)
            outputs = self.nn(input_tensor) # Logits
            probabilities = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probabilities, 1)

        idx = pred_idx.item()
        if not (0 <= idx < len(self.labels)):
            messagebox.showerror("Prediction Error", f"Predicted index {idx} out of bounds for labels (len {len(self.labels)}).")
            self.label_var.set("Error: Prediction index invalid.")
            self.last_input_tensor = None
            return

        lbl = self.labels[idx]
        self.last_input_tensor = input_tensor # Store for Confirm/Teach
        self.last_pred_idx = idx
        self.last_conf = conf.item()
        self.label_var.set(f"▶️ Predicted: {lbl} (Confidence: {self.last_conf:.2f})")
        
        self.btns['Confirm'].config(state='normal')
        self.btns['Teach'].config(state='normal')

    def _perform_online_update(self, input_tensor, true_idx):
        self.nn.train() # Set model to training mode
        self.optimizer.zero_grad()
        outputs = self.nn(input_tensor.to(self.device))
        loss = self.criterion(outputs, torch.LongTensor([true_idx]).to(self.device))
        loss.backward()
        self.optimizer.step()
        print(f"Online update for label '{self.labels[true_idx]}', Loss: {loss.item():.4f}")


    def confirm(self):
        if not hasattr(self, 'last_input_tensor') or self.last_input_tensor is None:
            messagebox.showinfo("Info","No valid prediction to confirm."); return
        
        true_idx = self.last_pred_idx
        self._perform_online_update(self.last_input_tensor, true_idx)
        
        # Record this confirmed sample
        self.records.append({
            'pil_image': self.pil_image_draw.copy().resize((50,50), Image.Resampling.LANCZOS),
            'x_tensor': self.last_input_tensor.squeeze(0).cpu(),
            'y_idx': true_idx,
            'true_label_name': self.labels[true_idx],
            'pred_label_name': self.labels[true_idx], 'correct':True, 'conf':self.last_conf
        })
        self.label_var.set(f"Confirmed '{self.labels[true_idx]}'. Model updated 👍")
        self.show_stats()
        self.clear()
        self.btns['Confirm'].config(state='disabled'); self.btns['Teach'].config(state='disabled')
        self.last_input_tensor = None

    def teach(self):
        if not hasattr(self, 'last_input_tensor') or self.last_input_tensor is None:
            messagebox.showinfo("Info","Predict first, then Teach."); return

        true_label_str = self._pick_hierarchical()
        if not true_label_str: return

        pred_label_str = self.labels[self.last_pred_idx]
        
        if true_label_str not in self.labels:
            self.labels.append(true_label_str)
            # Update CNN output layer and re-initialize optimizer for the new layer params
            self.nn.update_output_layer(len(self.labels), device=self.device)
            self.optimizer = optim.Adam(self.nn.parameters(), lr=self.optimizer.param_groups[0]['lr']) # Keep old LR
            print(f"Optimizer re-initialized for new output layer size: {len(self.labels)}")
            # Existing records' y_idx might need adjustment if label order changes, but here we append.

        true_label_idx = self.labels.index(true_label_str)
        self._perform_online_update(self.last_input_tensor, true_label_idx)
        
        self.records.append({
            'pil_image': self.pil_image_draw.copy().resize((50,50), Image.Resampling.LANCZOS),
            'x_tensor': self.last_input_tensor.squeeze(0).cpu(),
            'y_idx': true_label_idx,
            'true_label_name': true_label_str,'pred_label_name': pred_label_str,
            'correct': (true_label_str == pred_label_str),'conf':self.last_conf
        })
        self.label_var.set(f"Taught as '{true_label_str}'. Model updated 🚀")
        self.show_stats()
        self.clear()
        self.btns['Confirm'].config(state='disabled'); self.btns['Teach'].config(state='disabled')
        self.last_input_tensor = None

    def _pick_hierarchical(self): # Same as your provided, minor tweaks possible
        top = tk.Toplevel(self.master); top.title("Choose True Label Type")
        sel_type = tk.StringVar()
        def choose_type(t): sel_type.set(t); top.destroy()
        
        categories = ['Shape','Number','Letter','Other (Custom)']
        for cat_name in categories:
            tk.Button(top, text=cat_name, width=15, command=lambda cn=cat_name: choose_type(cn)).pack(padx=10,pady=5)
        
        top.grab_set(); self.master.wait_window(top)
        chosen_type = sel_type.get()
        if not chosen_type: return None

        final_label = tk.StringVar()
        if chosen_type in ('Shape','Number','Letter'):
            opts_map = {
              'Shape':['circle','square','triangle'],
              'Number':[str(i) for i in range(10)],
              'Letter':[chr(i) for i in range(ord('A'),ord('Z')+1)]
            }
            options = opts_map[chosen_type]
            top2 = tk.Toplevel(self.master); top2.title(f"Select {chosen_type}")
            def choose_option(opt): final_label.set(opt); top2.destroy()
            cols = 5 if chosen_type in ['Letter', 'Number'] else 3
            btn_frame = ttk.Frame(top2, padding="10"); btn_frame.pack(expand=True, fill=tk.BOTH)
            for i, opt_val in enumerate(options):
                tk.Button(btn_frame, text=opt_val, width=4, command=lambda ov=opt_val: choose_option(ov)
                ).grid(row=i//cols, column=i%cols, padx=3,pady=3, sticky="ew")
            top2.grab_set(); self.master.wait_window(top2)
        elif chosen_type == 'Other (Custom)':
            custom_label = simpledialog.askstring("Custom Label", "Enter the correct label:", parent=self.master)
            if custom_label and custom_label.strip(): final_label.set(custom_label.strip())
            else: return None
        return final_label.get() if final_label.get() else None

    def load_image_file(self):
        fn = filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
        if not fn: return
        try:
            loaded_pil_img = Image.open(fn).convert('L') # Grayscale
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not open image: {e}"); return
        
        # Display on canvas (scaled to canvas size)
        self.pil_image_draw = loaded_pil_img.resize((self.cs, self.cs), Image.Resampling.LANCZOS)
        # self.draw_tool is already associated with self.pil_image_draw, but re-assign if creating new Image
        self.draw_tool = ImageDraw.Draw(self.pil_image_draw) 

        tk_img = ImageTk.PhotoImage(self.pil_image_draw)
        self.canvas.delete('all')
        self.canvas.create_image(0,0,anchor='nw',image=tk_img)
        self.canvas.tk_img_ref = tk_img # Keep reference
        
        self.predict()

    def show_stats(self):
        # --- Text Statistics ---
        txt = "--- Class Statistics ---\n"
        class_counts = {lbl: 0 for lbl in self.labels}
        class_correct = {lbl: 0 for lbl in self.labels}
        
        for r in self.records:
            true_label = r.get('true_label_name')
            if true_label in class_counts:
                class_counts[true_label] += 1
                if r.get('correct') is True: class_correct[true_label] +=1
        
        for lbl in self.labels:
            N = class_counts[lbl]; C = class_correct[lbl]
            acc = (C/N)*100 if N > 0 else 0.0
            txt += f"{lbl:>12s} : {N:>3d} samples, acc {acc:>6.2f}%\n"
        
        confs = [r['conf'] for r in self.records if r.get('conf') is not None]
        avg_c = np.mean(confs) if confs else 0.0
        txt += f"\n--- Overall ---\nTotal predictions: {len(confs)}\nAvg confidence: {avg_c:.3f}\n"
        txt += f"Total samples: {len(self.records)}\nCurrent classes: {len(self.labels)}\n"
        
        self.stats_text_widget.delete('1.0','end')
        self.stats_text_widget.insert('end', txt)

        # --- Thumbnail Gallery ---
        for widget in self.scrollable_thumb_frame.winfo_children(): # Clear old thumbnails
            widget.destroy()
        
        if not self.records:
            tk.Label(self.scrollable_thumb_frame, text="No samples recorded yet.").pack()
            return

        cols = 4 # Number of thumbnails per row
        thumb_size = (60,60) # Thumbnail display size

        for i, r_idx in enumerate(range(len(self.records) -1, -1, -1)): # Show newest first
            record = self.records[r_idx]
            if 'pil_image' not in record: continue

            thumb_pil = record['pil_image'].copy() # Already resized to 50x50 during capture/confirm
            thumb_pil.thumbnail(thumb_size, Image.Resampling.LANCZOS) # Ensure it fits if original was larger
            
            tk_thumb = ImageTk.PhotoImage(thumb_pil)
            
            item_frame = tk.Frame(self.scrollable_thumb_frame, borderwidth=1, relief="sunken")
            item_frame.grid(row=i//cols, column=i%cols, padx=3, pady=3, sticky="nsew")

            thumb_label_widget = tk.Label(item_frame, image=tk_thumb)
            thumb_label_widget.image = tk_thumb # Keep a reference!
            thumb_label_widget.pack(pady=(0,2))

            info_text = f"True: {record.get('true_label_name', 'N/A')}"
            if record.get('pred_label_name') is not None:
                info_text += f"\nPred: {record.get('pred_label_name')}"
                col = "green" if record.get('correct') else "red"
                conf_val = record.get('conf',0)
                tk.Label(item_frame, text=f"Conf: {conf_val:.2f}", fg=col, font=('Arial', 7)).pack()

            tk.Label(item_frame, text=info_text, font=('Arial', 7)).pack()
        
        # Update scrollregion after adding all items
        self.scrollable_thumb_frame.update_idletasks()
        self.thumb_canvas.config(scrollregion=self.thumb_canvas.bbox("all"))


# --- Launch ---
if __name__ == "__main__":
    initial_labels_list = (
        ['circle','square','triangle']
        + [str(i) for i in range(10)]
        # + [chr(i) for i in range(ord('A'),ord('Z')+1)] # Add letters for more classes
    )
    # initial_labels_list = [] # To start completely fresh

    root = tk.Tk()
    root.title(f"📓 Drawing Learner (CNN v{APP_VERSION})")
    root.geometry("950x700") # Adjusted window size
    
    app = DrawApp(root, initial_labels=initial_labels_list, required=5) # required per class for initial phase
    
    root.mainloop()