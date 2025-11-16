# Logic: CFO (dùng giải thuật sắp xếp để tìm pixel có phương sai cao nhất 
# thay cho Swarm Intelligence trong bài báo)
#  + Khóa BSBE Pre-shared (Lần này Không nhúng vào metadata nữa mấy ný nhập tay cho nó bảo mật:)))

import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
import io
import os
import time

# ==================== CÁC HÀM TIỆN ÍCH (Cho CFO) ====================
def compute_variance_map(img, size=3):
    img = img.astype(float)
    mean = ndimage.uniform_filter(img, size)
    sq_mean = ndimage.uniform_filter(img**2, size)
    var_map = sq_mean - mean**2
    return np.clip(var_map, 0, None)

# ==================== THUẬT TOÁN CFO (Phiên bản ỔN ĐỊNH) ====================
def get_best_cfo_pixels(cover_gray, num_pixels):
    """
    Thực hiện đúng tinh thần của CFO (Bài báo):
    1. Tính bản đồ phương sai.
    2. Sắp xếp TẤT CẢ pixel theo phương sai (từ cao đến thấp).
    3. Trả về K pixel tốt nhất.
    Hàm này 100% ổn định (deterministic).
    """
    print("Đang tính toán bản đồ phương sai (Variance Map)...")
    start_time = time.time()
    
    var_map = compute_variance_map(cover_gray)
    h, w = cover_gray.shape
    
    print("Đang tạo danh sách pixel...")
    flat_indices = np.arange(h * w)
    flat_map = var_map.flatten()
    
    print("Đang sắp xếp các pixel theo phương sai (có thể mất vài giây)...")
    try:
        sorted_indices = flat_indices[np.argsort(flat_map)[::-1]]
    except MemoryError:
        print("⚠️ Cảnh báo bộ nhớ! Sử dụng phương pháp chậm hơn.")
        sorted_indices = sorted(flat_indices, key=lambda i: flat_map[i], reverse=True)
        
    best_indices = sorted_indices[:num_pixels]
    
    print("🔹 Đang chuyển đổi chỉ số sang tọa độ (x, y)...")
    pixel_coords = []
    for idx in best_indices:
        x = int(idx % w)
        y = int(idx // w)
        pixel_coords.append((x, y))
        
    end_time = time.time()
    print(f" Đã chọn {len(pixel_coords)} pixel tốt nhất (theo CFO) sau {end_time - start_time:.2f} giây.")
    
    if len(pixel_coords) < num_pixels:
        raise ValueError(f"Không thể chọn đủ {num_pixels} pixel. Ảnh bìa quá nhỏ?")
        
    return pixel_coords

# ==================== NÉN VÀ GIẢI NÉN ẢNH ====================
def compress_image_for_embedding(image_array, quality=85):
    img = Image.fromarray(image_array.astype('uint8'))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)
    compressed_size = buffer.getbuffer().nbytes
    print(f" Kích thước sau nén: {compressed_size} bytes")
    return buffer.getvalue()

def decompress_embedded_image(compressed_data):
    buffer = io.BytesIO(compressed_data)
    img = Image.open(buffer)
    return np.array(img)

# ==================== MÃ HÓA VÀ GIẢI MÃ BSBE ====================
# (Không thay đổi, 2 hàm này vốn đã nhận 'keys' từ bên ngoài)
def implement_bsbe_encryption(secret_image_array, keys):
    h, w, c = secret_image_array.shape
    Bx = By = 8
    encrypted = np.copy(secret_image_array).astype(np.uint8)
    for sx in range(0, h, Bx):
        for sy in range(0, w, By):
            ex, ey = min(sx + Bx, h), min(sy + By, w)
            block = encrypted[sx:ex, sy:ey]
            bh, bw = block.shape[:2]
            if bh != Bx or bw != By: continue
            
            np.random.seed(keys['K1'])
            rand_mat = np.random.randint(0, 256, block.shape, dtype=np.uint8)
            block ^= rand_mat
            
            np.random.seed(keys['K2'])
            perm = np.random.permutation(bh * bw)
            flat = block.reshape(-1, 3)[perm].reshape(bh, bw, 3)
            
            np.random.seed(keys['K3'])
            rands = np.random.randint(0, 2, bh * bw, dtype=np.uint8)
            flat = flat.reshape(-1, 3)
            flat[rands != 0] ^= 255
            
            np.random.seed(keys['K4'])
            perm_idx = np.random.randint(0, 6)
            perms = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
            flat = flat.reshape(bh, bw, 3)[:, :, perms[perm_idx]]
            encrypted[sx:ex, sy:ey] = flat
    print(" Đã mã hóa ảnh bí mật")
    return encrypted

def implement_bsbe_decryption(encrypted_array, keys):
    decrypted = np.copy(encrypted_array).astype(np.uint8)
    h, w, _ = decrypted.shape
    Bx = By = 8
    for sx in range(0, h, Bx):
        for sy in range(0, w, By):
            ex, ey = min(sx + Bx, h), min(sy + By, w)
            block = decrypted[sx:ex, sy:ey]
            bh, bw = block.shape[:2]
            if bh != Bx or bw != By: continue
            
            np.random.seed(keys['K4'])
            perm_idx = np.random.randint(0, 6)
            perms = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
            inv_perm = np.argsort(perms[perm_idx])
            block = block[:, :, inv_perm]
            
            np.random.seed(keys['K3'])
            num_pix = bh * bw
            rands = np.random.randint(0, 2, num_pix, dtype=np.uint8)
            flat = block.reshape(-1, 3)
            flat[rands != 0] ^= 255
            
            np.random.seed(keys['K2'])
            perm = np.random.permutation(num_pix)
            inv_perm = np.argsort(perm)
            flat = flat[inv_perm].reshape(bh, bw, 3)
            
            np.random.seed(keys['K1'])
            rand_mat = np.random.randint(0, 256, block.shape, dtype=np.uint8)
            block = flat ^ rand_mat
            decrypted[sx:ex, sy:ey] = block
    print(" Đã giải mã ảnh bí mật")
    return decrypted

# ==================== NHÚNG VÀ TRÍCH XUẤT METADATA (ĐÃ SỬA) ====================

# Metadata bây giờ CHỈ chứa kích thước file (4 bytes)
METADATA_SIZE_BYTES = 4 

def embed_metadata(stego_array, secret_size, pixel_coords):
    """Nhúng CHỈ kích thước file (4 bytes)"""
    metadata = bytearray()
    metadata.extend(secret_size.to_bytes(4, byteorder='big')) 
    
    for i in range(min(len(metadata), METADATA_SIZE_BYTES)):
        x, y = pixel_coords[i]
        byte_val = metadata[i]
        r_bits, g_bits, b_bits = (byte_val >> 5) & 0b111, (byte_val >> 2) & 0b111, byte_val & 0b11
        stego_array[y, x, 0] = (stego_array[y, x, 0] & 0b11111000) | r_bits
        stego_array[y, x, 1] = (stego_array[y, x, 1] & 0b11111000) | g_bits
        stego_array[y, x, 2] = (stego_array[y, x, 2] & 0b11111100) | b_bits
    return stego_array

def extract_metadata(stego_array, pixel_coords):
    """Trích xuất CHỈ kích thước file (4 bytes)"""
    metadata = bytearray()
    for i in range(METADATA_SIZE_BYTES):
        x, y = pixel_coords[i]
        pixel = stego_array[y, x]
        r_bits, g_bits, b_bits = pixel[0] & 0b111, pixel[1] & 0b111, pixel[2] & 0b11
        byte_val = (r_bits << 5) | (g_bits << 2) | b_bits
        metadata.append(byte_val)
    
    secret_size = int.from_bytes(metadata[0:4], byteorder='big')
    print(" Đã trích xuất metadata (chỉ kích thước file)")
    return secret_size # Không trả về keys nữa

# ==================== NHÚNG VÀ TRÍCH XUẤT DỮ LIỆU ====================
# (Đã cập nhật offset metadata)
def optimized_embedding(cover_array, secret_data, pixel_coords):
    stego = np.copy(cover_array)
    secret_flat = secret_data.flatten()
    required_pixels = len(secret_flat)
    available_pixels = len(pixel_coords) - METADATA_SIZE_BYTES # -4
    
    if required_pixels > available_pixels:
        error_msg = f"Dữ liệu quá lớn (cần {required_pixels} pixel, chỉ có {available_pixels} chỗ trống)"
        print(f"❌ LỖI NGHIÊM TRỌNG: {error_msg}")
        raise ValueError(error_msg)

    print(f" Đang nhúng {len(secret_flat)} bytes vào {available_pixels} pixel...")
    for i in range(required_pixels):
        x, y = pixel_coords[i + METADATA_SIZE_BYTES] # Offset 4
        secret_val = secret_flat[i]
        r_bits, g_bits, b_bits = (secret_val >> 5) & 0b111, (secret_val >> 2) & 0b111, secret_val & 0b11
        stego[y, x, 0] = (stego[y, x, 0] & 0b11111000) | r_bits
        stego[y, x, 1] = (stego[y, x, 1] & 0b11111000) | g_bits
        stego[y, x, 2] = (stego[y, x, 2] & 0b11111100) | b_bits
    return stego, len(secret_flat)

def extract_encrypted_data(stego_array, pixel_coords, original_compressed_size):
    # Dùng kích thước đệm (padding) lớn hơn cho ảnh 800x800
    temp_shape = (300, 300, 3) # 270,000 bytes
    padded_size = np.prod(temp_shape) 
    
    extracted_data = bytearray()
    available_pixels = len(pixel_coords) - METADATA_SIZE_BYTES # -4
    num_pixels_to_extract = min(padded_size, available_pixels)
    print(f" Đang trích xuất {num_pixels_to_extract} bytes (dữ liệu đã đệm)...")
    
    for i in range(num_pixels_to_extract):
        if i + METADATA_SIZE_BYTES >= len(pixel_coords): break 
        x, y = pixel_coords[i + METADATA_SIZE_BYTES] # Offset 4
        pixel = stego_array[y, x]
        r_bits, g_bits, b_bits = pixel[0] & 0b111, pixel[1] & 0b111, pixel[2] & 0b11
        byte_val = (r_bits << 5) | (g_bits << 2) | b_bits
        extracted_data.append(byte_val)
        
    if len(extracted_data) < padded_size:
        print(f" CẢNH BÁO: Trích xuất thiếu dữ liệu! {len(extracted_data)}/{padded_size} bytes.")
        extracted_data.extend(bytearray(padded_size - len(extracted_data)))

    data_array = np.frombuffer(extracted_data, dtype=np.uint8)
    print(f" Đã trích xuất {len(extracted_data)} bytes dữ liệu.")
    return data_array[:padded_size].reshape(temp_shape)

# ==================== LUỒNG CÔNG VIỆC CHÍNH (ĐÃ SỬA) ====================

def encode_image(cover_path, secret_path, keys, output_path="stego_image.png"):
    print(" Bắt đầu quá trình mã hóa...")
    cover_image = np.array(Image.open(cover_path).convert("RGB"))
    secret_image = np.array(Image.open(secret_path).convert("RGB"))
    
    print(f" Kích thước ảnh bìa: {cover_image.shape}")
    print(f" Kích thước ảnh bí mật: {secret_image.shape}")

    print(" Đang nén ảnh bí mật (để tính dung lượng)...")
    secret_compressed = compress_image_for_embedding(secret_image)
    original_compressed_size = len(secret_compressed)
    
    # Dùng kích thước đệm (padding) lớn hơn
    temp_shape = (300, 300, 3) # 270,000 bytes
    padded_size = np.prod(temp_shape)
    
    total_pixels_needed = padded_size + METADATA_SIZE_BYTES # +4
    print(f" Tổng dung lượng cần: {total_pixels_needed} pixel (cho metadata + data)")

    # Chạy CFO (Sắp xếp) để lấy danh sách pixel
    cover_gray = np.array(Image.open(cover_path).convert("L"))
    pixel_coords = get_best_cfo_pixels(cover_gray, total_pixels_needed)
    
    print(f" Đang sử dụng Khóa BSBE (pre-shared)...")
    
    print(" Đang mã hóa ảnh bí mật...")
    secret_array_for_encryption = np.frombuffer(secret_compressed, dtype=np.uint8)
    
    if len(secret_array_for_encryption) < padded_size:
        padded = np.zeros(padded_size, dtype=np.uint8)
        padded[:len(secret_array_for_encryption)] = secret_array_for_encryption
        secret_array_for_encryption = padded
    else:
        # Cắt bớt nếu file nén quá lớn
        secret_array_for_encryption = secret_array_for_encryption[:padded_size]
        original_compressed_size = padded_size # Cập nhật lại size
        print(f" CẢNH BÁO: Ảnh bí mật (đã nén) quá lớn, đã bị cắt còn {padded_size} bytes.")
        
    encrypted_secret = implement_bsbe_encryption(
        secret_array_for_encryption.reshape(temp_shape), keys
    )
    
    print(" Đang nhúng dữ liệu vào ảnh bìa...")
    stego, _ = optimized_embedding(
        cover_image, encrypted_secret, pixel_coords
    )
    
    print(" Đang nhúng metadata (chỉ kích thước)...")
    stego_with_metadata = embed_metadata(stego, original_compressed_size, pixel_coords)
    
    stego_img = Image.fromarray(stego_with_metadata.astype('uint8'))
    stego_img.save(output_path, "PNG") 
    
    print(f" Hoàn thành mã hóa!")
    print(f" Ảnh stego đã lưu tại: {output_path}")
    print(f" **THỨ QUAN TRỌNG CẦN GỬI:**")
    print(f"   1. File ảnh Stego: {os.path.basename(output_path)}")
    print(f"   2. File ảnh Bìa GỐC: {os.path.basename(cover_path)}")
    print(f"   (Người nhận PHẢI CÓ Khóa BSBE đã thống nhất)")

    return True

def decode_image(stego_path, original_cover_path, keys, output_path="recovered_secret.jpg"):
    print(" Bắt đầu quá trình giải mã...")
    stego_array = np.array(Image.open(stego_path))
    print(f" Kích thước ảnh stego: {stego_array.shape}")
    
    try:
        original_cover_gray = np.array(Image.open(original_cover_path).convert("L"))
    except FileNotFoundError:
        print(f" LỖI: Không tìm thấy ảnh bìa gốc tại: {original_cover_path}")
        raise
        
    print(f" Đã tải ảnh bìa gốc (để chạy CFO): {original_cover_gray.shape}")

    # Dùng kích thước đệm (padding) lớn hơn
    temp_shape = (300, 300, 3) # 270,000 bytes
    padded_size = np.prod(temp_shape)
    
    total_pixels_needed = padded_size + METADATA_SIZE_BYTES # +4
    
    # Chạy CFO (Sắp xếp) trên ảnh bìa GỐC để tạo lại ma trận
    pixel_coords = get_best_cfo_pixels(original_cover_gray, total_pixels_needed)
    
    print(" Đang trích xuất metadata (chỉ kích thước)...")
    secret_size = extract_metadata(stego_array, pixel_coords)
    print(f" Kích thước dữ liệu bí mật: {secret_size} bytes")
    
    print(" Đang trích xuất dữ liệu đã mã hóa...")
    encrypted_data = extract_encrypted_data(stego_array, pixel_coords, secret_size)
    
    print(f" Đang sử dụng Khóa BSBE (pre-shared) để giải mã...")
    decrypted_data = implement_bsbe_decryption(encrypted_data, keys)
    
    decrypted_bytes = decrypted_data.astype(np.uint8).tobytes()[:secret_size]
    
    print(" Đang giải nén ảnh bí mật...")
    recovered_image = decompress_embedded_image(decrypted_bytes)
    
    recovered_img = Image.fromarray(recovered_image.astype('uint8'))
    recovered_img.save(output_path, quality=95)
    
    print(f" Hoàn thành giải mã!")
    print(f" Ảnh bí mật đã khôi phục tại: {output_path}")
    
    return True