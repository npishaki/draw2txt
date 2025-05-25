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
        for i in range(epochs):
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

# --- Synthetic Shape Data for Pre-training ---
def make_shape_img(kind, size=28):
    img = np.zeros((size, size))
    rr, cc = np.ogrid[:size, :size]
    if kind == 0:  # circle
        mask = (rr - size//2)**2 + (cc - size//2)**2 < (size//3)**2
        img[mask] = 1
    elif kind == 1:  # square
        s = size//3
        img[size//2-s:size//2+s, size//2-s:size//2+s] = 1
    else:  # triangle
        for r in range(size//3, 2*size//3):
            c1 = size//2 - (r - size//3)
            c2 = size//2 + (r - size//3)
            img[r, c1:c2] = 1
    return img

def gen_shape_data(n=200):
    X, Y = [], []
    for k in range(3):
        for _ in range(n):
            im = make_shape_img(k)
            im += np.random.randn(*im.shape)*0.05
            X.append(im.flatten())
            vec = np.zeros(3); vec[k] = 1
            Y.append(vec)
    return np.array(X).T, np.array(Y).T

# --- Main Application ---
class DrawApp:
    def __init__(self, master, nn, labels):
        self.master = master
        self.nn = nn
        self.labels = labels[:]  # e.g. ["circle","square","triangle","0","1","A","B"]
        self.size, self.canvas_size = 28, 280
        self.bg, self.fg = 'white','black'

        # Records: each = {'x':…, 'y':…, 'img':…, 'true':str, 'pred':str, 'correct':bool}
        self.records = []
        self.last_conf = 0.0
        # --- Tabs ---
        self.nb = ttk.Notebook(master)
        self.draw_frame = tk.Frame(self.nb)
        self.stats_frame = tk.Frame(self.nb)
        self.nb.add(self.draw_frame, text="Draw")
        self.nb.add(self.stats_frame, text="Statistics")
        self.nb.pack(fill='both', expand=True)

        # --- DRAW TAB ---
        self.canvas = tk.Canvas(self.draw_frame, width=self.canvas_size,
                                height=self.canvas_size, bg=self.bg)
        self.canvas.grid(row=0, columnspan=6)
        self.image1 = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

        btns = [
            ("Clear", self.clear),
            ("Predict", self.predict),
            ("Confirm", self.confirm),
            ("Teach", self.teach),
            ("Load Image", self.load_image),
            ("Quit", master.quit)
        ]
        for i,(txt,cmd) in enumerate(btns):
            tk.Button(self.draw_frame, text=txt, width=10,
                      command=cmd).grid(row=1, column=i, padx=2, pady=4)

        self.label_var = tk.StringVar(
            value="Draw or Load→Predict→Confirm/Teach")
        tk.Label(self.draw_frame, textvariable=self.label_var,
                 font=('Arial',12)).grid(row=2, columnspan=6, pady=4)

        # --- STATS TAB ---
        self.stats_text = tk.Text(self.stats_frame, width=50, height=15)
        self.stats_text.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.stats_frame, text="Refresh Stats",
                  command=self.show_stats).pack(pady=4)
        # weight viz
        self.fig = plt.Figure(figsize=(4,4))
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.stats_frame)
        self.canvas_fig.get_tk_widget().pack(side=tk.RIGHT, padx=5)
        # reference images
        self.ref_scroll = tk.Canvas(self.stats_frame)
        self.ref_frame = tk.Frame(self.ref_scroll)
        self.vs = tk.Scrollbar(self.stats_frame, orient='vertical',
                               command=self.ref_scroll.yview)
        self.ref_scroll.configure(yscrollcommand=self.vs.set)
        self.vs.pack(side=tk.RIGHT, fill='y')
        self.ref_scroll.pack(side=tk.RIGHT, fill='both', expand=True)
        self.ref_scroll.create_window((0,0), window=self.ref_frame, anchor='nw')
        self.ref_frame.bind("<Configure>",
            lambda e: self.ref_scroll.configure(scrollregion=self.ref_scroll.bbox("all")))

        # --- Initial Stat Display ---
        self.show_stats()

    # --- Drawing/Canvas ---
    def paint(self, e):
        x1,y1 = e.x-8, e.y-8
        x2,y2 = e.x+8, e.y+8
        self.canvas.create_oval(x1,y1,x2,y2, fill=self.fg,outline=self.fg)
        self.draw.ellipse([x1,y1,x2,y2], fill=0)

    def clear(self):
        self.canvas.delete('all')
        self.draw.rectangle([0,0,self.canvas_size,self.canvas_size], fill=255)
        self.label_var.set("Canvas cleared. Draw or load an image.")

    def get_nn_input(self):
        img = self.image1.resize((self.size,self.size),
                                 Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        arr = np.array(img)/255.0
        return arr.flatten()[:,None]

    # --- Prediction Flow ---
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
            messagebox.showinfo("Info","No prediction to confirm.")
            return
        lbl = self.labels[self.last_pred_idx]
        Y = np.zeros((len(self.labels),1)); Y[self.last_pred_idx]=1
        self.nn.online_update(self.last_X, Y)
        self.records.append({
            'x':self.last_X, 'y':Y,
            'img':self.image1.copy(),
            'true':lbl, 'pred':lbl, 'correct':True,
            'conf':self.last_conf
        })
        self.label_var.set(f"Confirmed '{lbl}'. Learned 👍")
        self.show_stats()

    def teach(self):
        if not hasattr(self, 'last_X'):
            messagebox.showinfo("Info","Predict first, then Teach.")
            return
        true_lbl = self.pick_label_hierarchical()
        if not true_lbl: return
        pred_lbl = self.labels[self.last_pred_idx]
        # expand labels/model if new
        if true_lbl not in self.labels:
            self.labels.append(true_lbl)
            OD = len(self.labels)
            newNN = SimpleNN(self.size*self.size,64,OD)
            # copy old
            newNN.W1[:,:] = self.nn.W1
            newNN.b1[:]   = self.nn.b1
            newNN.W2[:self.nn.W2.shape[0],:] = self.nn.W2
            newNN.b2[:self.nn.b2.shape[0],:]   = self.nn.b2
            self.nn = newNN
        Y = np.zeros((len(self.labels),1)); Y[self.labels.index(true_lbl)] = 1
        self.nn.online_update(self.last_X, Y)
        self.records.append({
            'x':self.last_X, 'y':Y,
            'img':self.image1.copy(),
            'true':true_lbl,'pred':pred_lbl,
            'correct':False,'conf':self.last_conf
        })
        self.label_var.set(f"Taught as '{true_lbl}'. Learned 🚀")
        self.show_stats()

    def pick_label_hierarchical(self):
        top = tk.Toplevel(self.master); top.title("Choose Type")
        sel = tk.StringVar()
        def choose(t):
            sel.set(t); top.destroy()
        for t in ('Shape','Number','Letter','Other'):
            tk.Button(top, text=t,width=12,
                      command=lambda t=t: choose(t)).pack(padx=5,pady=3)
        top.grab_set(); top.wait_window()
        tp = sel.get()
        if tp in ('Shape','Number','Letter'):
            opts = {
              'Shape':['circle','square','triangle','star','heart'],
              'Number':[str(i) for i in range(10)],
              'Letter':[chr(i) for i in range(65,91)]
            }[tp]
            top2 = tk.Toplevel(self.master); top2.title(tp)
            sel2 = tk.StringVar()
            frm = tk.Frame(top2); frm.pack(padx=5,pady=5)
            for i,opt in enumerate(opts):
                tk.Button(frm, text=opt,width=4,
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
        self.image1 = im.resize(
            (self.canvas_size,self.canvas_size),
            Image.Resampling.LANCZOS)
        self.draw = ImageDraw.Draw(self.image1)
        self.tkimg = ImageTk.PhotoImage(self.image1)
        self.canvas.create_image(0,0,anchor='nw',image=self.tkimg)
        self.predict()

    # --- Stats & Prune ---
    def show_stats(self):
        # compute per-class counts & accuracy
        txt = ""
        for lbl in self.labels:
            recs = [r for r in self.records if r['true']==lbl]
            N = len(recs)
            C = sum(r['correct'] for r in recs)
            acc = C/N if N>0 else 0
            txt += f"{lbl:>8} : {N:>3} samples, acc {acc:>5.2f}\n"
        # avg confidence
        confs = [r['conf'] for r in self.records]
        avg_c = np.mean(confs) if confs else 0
        txt += f"\nAvg confidence: {avg_c:.2f}\n"
        self.stats_text.delete('1.0','end')
        self.stats_text.insert('end', txt)

        # visualize weights
        self.fig.clf()
        G = int(np.ceil(np.sqrt(len(self.labels))))
        for i,lbl in enumerate(self.labels):
            ax = self.fig.add_subplot(G,G,i+1)
            w = self.nn.W2[i] @ self.nn.W1   # (in_dim,)
            try:
                im = w.reshape(self.size,self.size)
                ax.imshow(im, cmap='gray')
            except:
                pass
            ax.set_title(lbl); ax.axis('off')
        self.canvas_fig.draw()

        # show reference images
        for w in self.ref_frame.winfo_children():
            w.destroy()
        for idx,r in enumerate(self.records):
            thumb = r['img'].resize((50,50),
                       Image.Resampling.LANCZOS)
            tkimg = ImageTk.PhotoImage(thumb)
            lbl = tk.Label(self.ref_frame,image=tkimg,
                           bd=1,relief='solid')
            lbl.image = tkimg
            lbl.grid(row=idx//10, column=idx%10,
                     padx=2,pady=2)
            lbl.bind('<Button-1>',
                     lambda e,i=idx: self.prune_image(i))

    def prune_image(self, idx):
        if not messagebox.askyesno(
            "Prune","Remove this sample?"): return
        self.records.pop(idx)
        # rebuild model: pretrain shapes + re-apply all records
        L = len(self.labels)
        self.nn = SimpleNN(self.size*self.size,64,L)
        # pretrain shapes
        Xs,Ys = gen_shape_data(200)
        Ys_full = np.zeros((L, Ys.shape[1]))
        Ys_full[:3,:] = Ys
        self.nn.train(Xs, Ys_full, lr=0.2, epochs=200)
        # replay records
        for r in self.records:
            self.nn.online_update(r['x'], r['y'], lr=0.4)
        self.show_stats()

# --- Launch ---
if __name__ == "__main__":
    # initial classes & model
    labels = ["circle","square","triangle","0","1","A","B"]
    nn = SimpleNN(28*28, 64, len(labels))
    # pretrain basic shapes
    X0,Y0 = gen_shape_data(200)
    Y0_full = np.zeros((len(labels), Y0.shape[1]))
    Y0_full[:3,:] = Y0
    nn.train(X0, Y0_full, lr=0.2, epochs=200)

    root = tk.Tk()
    root.title("📓 Handwriting & Shape Online Learner")
    app = DrawApp(root, nn, labels)
    root.mainloop()
