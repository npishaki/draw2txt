import numpy as np
import tkinter as tk
from tkinter import simpledialog, ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Simple Feed-Forward Network with Online Update ---
class SimpleNN:
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.W1 = np.random.randn(hidden_dim, in_dim) * 0.01
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(out_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((out_dim, 1))

    def relu(self, z): return np.maximum(0, z)
    def relu_deriv(self, z): return (z > 0).astype(float)

    def softmax(self, z):
        ex = np.exp(z - np.max(z, axis=0, keepdims=True))
        return ex / ex.sum(axis=0, keepdims=True)

    def forward(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def backward(self, X, Y, Z1, A1, A2):
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = (dZ2 @ A1.T) / m
        db2 = dZ2.sum(axis=1, keepdims=True) / m
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * self.relu_deriv(Z1)
        dW1 = (dZ1 @ X.T) / m
        db1 = dZ1.sum(axis=1, keepdims=True) / m
        return dW1, db1, dW2, db2

    def train(self, X, Y, lr=0.2, epochs=200):
        for _ in range(epochs):
            Z1, A1, Z2, A2 = self.forward(X)
            dW1, db1, dW2, db2 = self.backward(X, Y, Z1, A1, A2)
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

    def online_update(self, X, Y, lr=0.4):
        Z1, A1, Z2, A2 = self.forward(X)
        dW1, db1, dW2, db2 = self.backward(X, Y, Z1, A1, A2)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return np.argmax(A2, axis=0), A2

# --- Main Application ---
class DrawApp:
    def __init__(self, master, nn, labels, required=10):
        self.master = master
        self.nn = nn
        self.labels = labels[:]        # initial classes
        self.required = required       # examples per class
        self.size, self.cs = 28, 280
        self.bg, self.fg = 'white','black'
        self.records = []              # store all samples
        self.last_conf = 0.0

        # --- Notebook and Frames ---
        self.nb = ttk.Notebook(master)
        self.draw_f = tk.Frame(self.nb)
        self.stats_f = tk.Frame(self.nb)
        self.nb.add(self.draw_f, text="Draw")
        self.nb.add(self.stats_f, text="Statistics")
        self.nb.pack(fill='both', expand=True)

        # --- Draw Tab ---
        self.canvas = tk.Canvas(self.draw_f, width=self.cs,
                                height=self.cs, bg=self.bg)
        self.canvas.grid(row=0, columnspan=7)
        self.image1 = Image.new('L', (self.cs, self.cs), 255)
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

        # Buttons
        cmds = [
            ("Clear", self.clear),
            ("Capture", self.capture_sample),
            ("Predict", self.predict),
            ("Confirm", self.confirm),
            ("Teach", self.teach),
            ("Load Image", self.load_image),
            ("Quit", master.quit),
        ]
        self.btns = {}
        for i,(txt,cmd) in enumerate(cmds):
            b = tk.Button(self.draw_f, text=txt, width=10, command=cmd)
            b.grid(row=1, column=i, padx=2, pady=4)
            self.btns[txt] = b

        self.label_var = tk.StringVar(
            value="— Initializing: please supply samples —"
        )
        tk.Label(self.draw_f, textvariable=self.label_var,
                 font=('Arial',12)).grid(row=2, columnspan=7, pady=4)

        # --- Stats Tab ---
        self.stats_text = tk.Text(self.stats_f, width=50, height=15)
        self.stats_text.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.stats_f, text="Refresh Stats",
                  command=self.show_stats).pack(pady=4)
        self.fig = plt.Figure(figsize=(4,4))
        self.canvas_fig = FigureCanvasTkAgg(self.fig,
                                            master=self.stats_f)
        self.canvas_fig.get_tk_widget().pack(side=tk.RIGHT,
                                             padx=5)

        # --- Begin Initial Sampling ---
        self._init_label_idx = 0
        self._init_count = 0
        self._set_initial_mode(True)
        self._collect_initial_samples()

    def _set_initial_mode(self, initial):
        if initial:
            # only Capture, Clear, Quit
            for name, btn in self.btns.items():
                btn.config(state='normal' if name in ('Capture','Clear','Quit') else 'disabled')
        else:
            # unlock everything except Capture
            for name, btn in self.btns.items():
                btn.config(state='disabled' if name=='Capture' else 'normal')

    def _collect_initial_samples(self):
        if self._init_label_idx >= len(self.labels):
            # done
            self.label_var.set("✅ Initial samples collected! You may now Predict.")
            self._set_initial_mode(False)
            # batch-train on your samples
            X = np.hstack([r['x'] for r in self.records])
            Y = np.hstack([r['y'] for r in self.records])
            self.nn.train(X, Y, lr=0.2, epochs=300)
            self.show_stats()
            return

        lbl = self.labels[self._init_label_idx]
        n   = self._init_count
        self.label_var.set(
            f"Draw '{lbl}' — sample {n+1}/{self.required}, then Capture."
        )

    def paint(self, e):
        x1,y1 = e.x-8, e.y-8
        x2,y2 = e.x+8, e.y+8
        self.canvas.create_oval(x1,y1,x2,y2,
                                fill=self.fg,outline=self.fg)
        self.draw.ellipse([x1,y1,x2,y2], fill=0)

    def clear(self):
        self.canvas.delete('all')
        self.draw.rectangle([0,0,self.cs,self.cs], fill=255)

    def get_nn_input(self):
        img = self.image1.resize((self.size,self.size),
                                 Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        arr = np.array(img)/255.0
        return arr.flatten()[:,None]

    # --- INITIAL CAPTURE ---
    def capture_sample(self):
        X = self.get_nn_input()
        y = np.zeros((len(self.labels),1))
        y[self._init_label_idx,0] = 1
        self.records.append({
            'x':X, 'y':y,
            'img':self.image1.copy(),
            'true':self.labels[self._init_label_idx],
            'pred':None, 'correct':True, 'conf':None
        })
        self._init_count += 1
        self.clear()
        if self._init_count >= self.required:
            self._init_count = 0
            self._init_label_idx += 1
        self._collect_initial_samples()

    # --- NORMAL WORKFLOW ---
    def predict(self):
        X = self.get_nn_input()
        pred, probs = self.nn.predict(X)
        idx = int(pred.item())
        lbl = self.labels[idx]
        conf = float(np.max(probs))
        self.last_X, self.last_pred_idx = X, idx
        self.last_conf = conf
        self.label_var.set(f"▶️ {lbl}  (conf {conf:.2f})")

    def confirm(self):
        if not hasattr(self, 'last_X'):
            messagebox.showinfo("Info","No prediction to confirm."); return
        idx = self.last_pred_idx
        Y = np.zeros((len(self.labels),1)); Y[idx]=1
        self.nn.online_update(self.last_X, Y)
        self.records.append({
            'x':self.last_X, 'y':Y,
            'img':self.image1.copy(),
            'true':self.labels[idx],
            'pred':self.labels[idx],
            'correct':True,'conf':self.last_conf
        })
        self.label_var.set(f"Confirmed '{self.labels[idx]}'. Learned 👍")
        self.show_stats()

    def teach(self):
        if not hasattr(self, 'last_X'):
            messagebox.showinfo("Info","Predict first, then Teach."); return
        true_lbl = self._pick_hierarchical()
        if not true_lbl: return
        pred_lbl = self.labels[self.last_pred_idx]
        if true_lbl not in self.labels:
            self.labels.append(true_lbl)
            OD = len(self.labels)
            newNN = SimpleNN(self.size*self.size,64,OD)
            newNN.W1[:,:] = self.nn.W1
            newNN.b1[:]   = self.nn.b1
            newNN.W2[:self.nn.W2.shape[0],:] = self.nn.W2
            newNN.b2[:self.nn.b2.shape[0],:] = self.nn.b2
            self.nn = newNN
        Y = np.zeros((len(self.labels),1))
        Y[self.labels.index(true_lbl)] = 1
        self.nn.online_update(self.last_X, Y)
        self.records.append({
            'x':self.last_X,'y':Y,
            'img':self.image1.copy(),
            'true':true_lbl,'pred':pred_lbl,
            'correct':False,'conf':self.last_conf
        })
        self.label_var.set(f"Taught as '{true_lbl}'. Learned 🚀")
        self.show_stats()

    def _pick_hierarchical(self):
        top = tk.Toplevel(self.master); top.title("Choose Type")
        sel = tk.StringVar()
        def choose(t): sel.set(t); top.destroy()
        for t in ('Shape','Number','Letter','Other'):
            tk.Button(top, text=t, width=12,
                      command=lambda t=t: choose(t)).pack(padx=5,pady=3)
        top.grab_set(); top.wait_window()
        tp = sel.get()
        if tp in ('Shape','Number','Letter'):
            opts = {
              'Shape':['circle','square','triangle'],
              'Number':[str(i) for i in range(10)],
              'Letter':[chr(i) for i in range(65,91)]
            }[tp]
            top2 = tk.Toplevel(self.master); top2.title(tp)
            sel2 = tk.StringVar()
            frm = tk.Frame(top2); frm.pack(padx=5,pady=5)
            for i,opt in enumerate(opts):
                tk.Button(frm, text=opt, width=4,
                  command=lambda o=opt: (sel2.set(o), top2.destroy())
                ).grid(row=i//10, column=i%10, padx=2,pady=2)
            top2.grab_set(); top2.wait_window()
            return sel2.get().strip()
        else:
            txt = simpledialog.askstring("Other","Enter label:")
            return txt.strip() if txt else None

    def load_image(self):
        fn = filedialog.askopenfilename(
            filetypes=[("Images","*.png;*.jpg;*.bmp")])
        if not fn: return
        im = Image.open(fn).convert('L')
        self.image1 = im.resize((self.cs,self.cs),
                                Image.Resampling.LANCZOS)
        self.draw = ImageDraw.Draw(self.image1)
        self.tkimg = ImageTk.PhotoImage(self.image1)
        self.canvas.create_image(0,0,anchor='nw',image=self.tkimg)
        self.predict()

    # --- Statistics & Pruning (same as before) ---
    def show_stats(self):
        txt = ""
        for lbl in self.labels:
            recs = [r for r in self.records if r['true']==lbl]
            N = len(recs)
            C = sum(r['correct'] for r in recs)
            acc = (C/N) if N>0 else 0
            txt += f"{lbl:>8} : {N:>3} samples, acc {acc:>5.2f}\n"
        confs = [r['conf'] for r in self.records if r['conf']!=None]
        avg_c = np.mean(confs) if confs else 0
        txt += f"\nAvg confidence: {avg_c:.2f}\n"
        self.stats_text.delete('1.0','end')
        self.stats_text.insert('end', txt)

        self.fig.clf()
        G = int(np.ceil(np.sqrt(len(self.labels))))
        for i,lbl in enumerate(self.labels):
            ax = self.fig.add_subplot(G,G,i+1)
            w = self.nn.W2[i] @ self.nn.W1
            try:
                im = w.reshape(self.size,self.size)
                ax.imshow(im, cmap='gray')
            except: pass
            ax.set_title(lbl); ax.axis('off')
        self.canvas_fig.draw()

# --- Launch ---
if __name__ == "__main__":
    # shapes + numbers + uppercase letters
    labels = (
        ['circle','square','triangle']
        + [str(i) for i in range(10)]
        + [chr(i) for i in range(65,91)]
    )
    nn = SimpleNN(28*28, 64, len(labels))
    root = tk.Tk()
    root.title("📓 Handwriting & Shape Learner")
    app = DrawApp(root, nn, labels, required=10)
    root.mainloop()
