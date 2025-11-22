import io
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

# 定义与加密脚本相同的密钥
# !!! 警告：请务必将此密钥更改为强密码，并妥善保管 !!!
# 密钥长度必须是 16, 24 或 32 字节
SECRET_KEY = b'0123456789abcdef0123456789abcdef'

def decrypt_file_content(encrypted_file_path, key):
    """
    使用 AES 解密文件内容并返回字节流。
    """
    try:
        with open(encrypted_file_path, 'rb') as f:
            iv = f.read(16)  # 读取初始化向量
            ct = f.read()    # 读取密文
        
        cipher = AES.new(key, AES.MODE_CBC, iv)
        # 解密并移除填充
        data = unpad(cipher.decrypt(ct), AES.block_size)
        return io.BytesIO(data)
    except FileNotFoundError:
        print(f"❌ 错误: 未找到加密文件 {encrypted_file_path}，请确认路径是否正确。")
        return None
    except Exception as e:
        print(f"❌ 解密文件 {encrypted_file_path} 时出错: {e}")
        return None 