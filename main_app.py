import tkinter as tk
from tkinter import ttk
import app_components 

# Biến toàn cục để theo dõi cửa sổ
sender_window = None
receiver_window = None

def open_sender_window():
    """Mở cửa sổ Mã hóa (Người Gửi)"""
    global sender_window
    # Chỉ cho phép mở 1 cửa sổ Gửi mỗi lần
    if sender_window and sender_window.winfo_exists():
        sender_window.lift()
        return

    # Tạo một cửa sổ Toplevel (cửa sổ con)
    sender_window = tk.Toplevel(root)
    sender_window.title("Công cụ Mã hóa (Người Gửi)")
    sender_window.geometry("550x500")
    
    # "Vẽ" giao diện người gửi vào cửa sổ con này
    app_components.build_sender_ui(sender_window)

def open_receiver_window():
    """Mở cửa sổ Giải mã (Người Nhận)"""
    global receiver_window
    # Chỉ cho phép mở 1 cửa sổ Nhận mỗi lần
    if receiver_window and receiver_window.winfo_exists():
        receiver_window.lift()
        return
        
    # Tạo một cửa sổ Toplevel (cửa sổ con)
    receiver_window = tk.Toplevel(root)
    receiver_window.title("Công cụ Giải mã (Người Nhận)")
    receiver_window.geometry("550x500")
    
    # "Vẽ" giao diện người nhận vào cửa sổ con này
    app_components.build_receiver_ui(receiver_window)


# --- Tạo Cửa sổ Chính (Menu) ---
root = tk.Tk()
root.title("Bảng điều khiển chính - CFOPS-BSBEA")
root.geometry("400x250")

main_frame = ttk.Frame(root, padding=20)
main_frame.pack(expand=True, fill=tk.BOTH)

ttk.Label(main_frame, text="CHỌN CHỨC NĂNG", font=("Helvetica", 16, "bold")).pack(pady=10)

# Nút Giấu tin
btn_sender = ttk.Button(main_frame, text="Giấu tin (Mã hóa)", command=open_sender_window)
btn_sender.pack(pady=10, fill=tk.X, ipady=10)

# Nút Giải mã
btn_receiver = ttk.Button(main_frame, text="Giải mã (Trích xuất)", command=open_receiver_window)
btn_receiver.pack(pady=10, fill=tk.X, ipady=10)

# Nút Thoát
btn_exit = ttk.Button(main_frame, text="Thoát", command=root.destroy)
btn_exit.pack(pady=10, fill=tk.X, ipady=10)

# Chạy vòng lặp chính
root.mainloop()