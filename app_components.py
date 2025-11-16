import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import core_logic  # Import file logic CFO
import threading
import os


# ===== LOGIC VÀ GIAO DIỆN CHO NGƯỜI GỬI (SENDER) =====

def start_encoding_thread(cover_path, secret_path, keys, output_path, status_label, root):
    """Hàm chạy mã hóa trong một thread riêng"""
    try:
        status_label.config(text="Đang chạy CFO (Sắp xếp) và Mã hóa... (có thể mất 10-20 giây)")
        root.update_idletasks()
        
        core_logic.encode_image(cover_path, secret_path, keys, output_path)
        
        messagebox.showinfo(
            "Hoàn thành",
            f"Đã mã hóa và lưu ảnh thành công!\n\n"
            f"File đã lưu tại: {output_path}\n\n"
            f"BÂY GIỜ BẠN CẦN GỬI 2 THỨ CHO NGƯỜI NHẬN:\n"
            f"1. File ảnh Stego: {os.path.basename(output_path)}\n"
            f"2. File ảnh Bìa GỐC: {os.path.basename(cover_path)}\n\n"
            f"(Người nhận phải tự biết 4 Khóa BSBE bạn vừa nhập)"
        )
        status_label.config(text="Sẵn sàng.")
        
    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")
        status_label.config(text="Đã xảy ra lỗi!")

def build_sender_ui(parent_window):
    """Hàm này "vẽ" toàn bộ giao diện Người Gửi vào cửa sổ cha"""
    
    main_frame = ttk.Frame(parent_window, padding="10")
    main_frame.pack(expand=True, fill=tk.BOTH)

    cover_path = tk.StringVar()
    secret_path = tk.StringVar()
    k1_var = tk.StringVar()
    k2_var = tk.StringVar()
    k3_var = tk.StringVar()
    k4_var = tk.StringVar()

    # --- Chọn Ảnh Bìa ---
    ttk.Label(main_frame, text="1. Chọn ảnh bìa (Cover Image):").pack(anchor=tk.W)
    cover_frame = ttk.Frame(main_frame)
    cover_frame.pack(fill=tk.X, pady=5)
    ttk.Entry(cover_frame, textvariable=cover_path, width=50).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
    ttk.Button(cover_frame, text="Duyệt...", command=lambda: cover_path.set(filedialog.askopenfilename(
        title="Chọn ảnh bìa", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    ))).pack(side=tk.RIGHT)

    # --- Chọn Ảnh Bí Mật ---
    ttk.Label(main_frame, text="2. Chọn ảnh bí mật (Secret Image):").pack(anchor=tk.W, pady=(10, 0))
    secret_frame = ttk.Frame(main_frame)
    secret_frame.pack(fill=tk.X, pady=5)
    ttk.Entry(secret_frame, textvariable=secret_path, width=50).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
    ttk.Button(secret_frame, text="Duyệt...", command=lambda: secret_path.set(filedialog.askopenfilename(
        title="Chọn ảnh bí mật", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    ))).pack(side=tk.RIGHT)

    # --- Nhập 4 Khóa BSBE ---
    ttk.Label(main_frame, text="3. Nhập 4 Khóa BSBE (phải là số, thống nhất trước):").pack(anchor=tk.W, pady=(10, 0))
    keys_frame = ttk.Frame(main_frame)
    keys_frame.pack(fill=tk.X, pady=5)
    ttk.Label(keys_frame, text="K1:").pack(side=tk.LEFT, padx=(0,5))
    ttk.Entry(keys_frame, textvariable=k1_var, width=8, show="*").pack(side=tk.LEFT, padx=5)
    ttk.Label(keys_frame, text="K2:").pack(side=tk.LEFT, padx=(10,5))
    ttk.Entry(keys_frame, textvariable=k2_var, width=8, show="*").pack(side=tk.LEFT, padx=5)
    ttk.Label(keys_frame, text="K3:").pack(side=tk.LEFT, padx=(10,5))
    ttk.Entry(keys_frame, textvariable=k3_var, width=8, show="*").pack(side=tk.LEFT, padx=5)
    ttk.Label(keys_frame, text="K4:").pack(side=tk.LEFT, padx=(10,5))
    ttk.Entry(keys_frame, textvariable=k4_var, width=8, show="*").pack(side=tk.LEFT, padx=5)

    # --- Nút Mã Hóa ---
    ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=15)
    status_label = ttk.Label(main_frame, text="Sẵn sàng.", relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

    def on_encode_click():
        if not cover_path.get() or not secret_path.get():
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn cả ảnh bìa và ảnh bí mật.")
            return
        try:
            keys = {
                'K1': int(k1_var.get()), 'K2': int(k2_var.get()),
                'K3': int(k3_var.get()), 'K4': int(k4_var.get())
            }
        except ValueError:
            messagebox.showwarning("Khóa không hợp lệ", "Vui lòng nhập 4 Khóa BSBE (phải là SỐ NGUYÊN).")
            return
        output_path = filedialog.asksaveasfilename(
            title="Lưu ảnh Stego", defaultextension=".png", filetypes=[("PNG Image", "*.png")]
        )
        if not output_path: return 
        threading.Thread(
            target=start_encoding_thread, 
            args=(cover_path.get(), secret_path.get(), keys, output_path, status_label, parent_window),
            daemon=True
        ).start()

    ttk.Button(main_frame, text="BẮT ĐẦU MÃ HÓA VÀ LƯU", command=on_encode_click).pack(pady=10, ipady=10, fill=tk.X)


# ===== LOGIC VÀ GIAO DIỆN CHO NGƯỜI NHẬN (RECEIVER) =====


def start_decoding_thread(stego_path, original_cover_path, keys, output_path, status_label, root):
    """Hàm chạy giải mã trong một thread riêng"""
    try:
        status_label.config(text="Đang chạy CFO (Sắp xếp) và Giải mã... (có thể mất 10-20 giây)")
        root.update_idletasks()
        
        core_logic.decode_image(stego_path, original_cover_path, keys, output_path)
        
        messagebox.showinfo(
            "Hoàn thành",
            f"Đã giải mã và khôi phục ảnh thành công!\n\n"
            f"File đã lưu tại: {output_path}"
        )
        status_label.config(text="Sẵn sàng.")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")
        status_label.config(text="Đã xảy ra lỗi!")

def build_receiver_ui(parent_window):
    """Hàm này "vẽ" toàn bộ giao diện Người Nhận vào cửa sổ cha"""

    main_frame = ttk.Frame(parent_window, padding="10")
    main_frame.pack(expand=True, fill=tk.BOTH)

    stego_path = tk.StringVar()
    original_cover_path = tk.StringVar()
    k1_var = tk.StringVar()
    k2_var = tk.StringVar()
    k3_var = tk.StringVar()
    k4_var = tk.StringVar()

    # --- Chọn Ảnh Stego ---
    ttk.Label(main_frame, text="1. Chọn ảnh Stego (Ảnh đã mã hóa):").pack(anchor=tk.W)
    stego_frame = ttk.Frame(main_frame)
    stego_frame.pack(fill=tk.X, pady=5)
    ttk.Entry(stego_frame, textvariable=stego_path, width=50).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
    ttk.Button(stego_frame, text="Duyệt...", command=lambda: stego_path.set(filedialog.askopenfilename(
        title="Chọn ảnh Stego", filetypes=[("PNG Image", "*.png")]
    ))).pack(side=tk.RIGHT)

    # --- Chọn Ảnh Bìa GỐC ---
    ttk.Label(main_frame, text="2. Chọn ảnh Bìa GỐC (Người gửi cung cấp):").pack(anchor=tk.W, pady=(10, 0))
    cover_frame = ttk.Frame(main_frame)
    cover_frame.pack(fill=tk.X, pady=5)
    ttk.Entry(cover_frame, textvariable=original_cover_path, width=50).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
    ttk.Button(cover_frame, text="Duyệt...", command=lambda: original_cover_path.set(filedialog.askopenfilename(
        title="Chọn ảnh bìa GỐC", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    ))).pack(side=tk.RIGHT)

    # --- Nhập 4 Khóa BSBE ---
    ttk.Label(main_frame, text="3. Nhập 4 Khóa BSBE (đã thống nhất):").pack(anchor=tk.W, pady=(10, 0))
    keys_frame = ttk.Frame(main_frame)
    keys_frame.pack(fill=tk.X, pady=5)
    ttk.Label(keys_frame, text="K1:").pack(side=tk.LEFT, padx=(0,5))
    ttk.Entry(keys_frame, textvariable=k1_var, width=8, show="*").pack(side=tk.LEFT, padx=5)
    ttk.Label(keys_frame, text="K2:").pack(side=tk.LEFT, padx=(10,5))
    ttk.Entry(keys_frame, textvariable=k2_var, width=8, show="*").pack(side=tk.LEFT, padx=5)
    ttk.Label(keys_frame, text="K3:").pack(side=tk.LEFT, padx=(10,5))
    ttk.Entry(keys_frame, textvariable=k3_var, width=8, show="*").pack(side=tk.LEFT, padx=5)
    ttk.Label(keys_frame, text="K4:").pack(side=tk.LEFT, padx=(10,5))
    ttk.Entry(keys_frame, textvariable=k4_var, width=8, show="*").pack(side=tk.LEFT, padx=5)

    # --- Nút Giải Mã ---
    ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=15)
    status_label = ttk.Label(main_frame, text="Sẵn sàng.", relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

    def on_decode_click():
        if not stego_path.get() or not original_cover_path.get():
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn cả ảnh Stego và ảnh Bìa GỐC.")
            return
        try:
            keys = {
                'K1': int(k1_var.get()), 'K2': int(k2_var.get()),
                'K3': int(k3_var.get()), 'K4': int(k4_var.get())
            }
        except ValueError:
            messagebox.showwarning("Khóa không hợp lệ", "Vui lòng nhập 4 Khóa BSBE (phải là SỐ NGUYÊN).")
            return
        output_path = filedialog.asksaveasfilename(
            title="Lưu ảnh đã khôi phục", defaultextension=".jpg", filetypes=[("JPEG Image", "*.jpg")]
        )
        if not output_path: return 
        threading.Thread(
            target=start_decoding_thread, 
            args=(stego_path.get(), original_cover_path.get(), keys, output_path, status_label, parent_window),
            daemon=True
        ).start()

    ttk.Button(main_frame, text="BẮT ĐẦU GIẢI MÃ VÀ LƯU", command=on_decode_click).pack(pady=10, ipady=10, fill=tk.X)