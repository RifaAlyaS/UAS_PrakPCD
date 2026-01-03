from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

def encode_image(img):
    """Encode image (RGB or GRAY) to base64 PNG.
    If input is RGB, convert to BGR before cv2.imencode to preserve true colors.
    """
    # Convert RGB to BGR for correct encoding in OpenCV
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_to_encode = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_to_encode = img

    success, buffer = cv2.imencode('.png', img_to_encode)
    if not success:
        raise ValueError('Failed to encode image')
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def plot_histogram(arr: np.ndarray, title: str) -> str:
    """Plot histogram as PNG base64 for a single-channel array."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.hist(arr.ravel(), bins=256, range=(0, 255), color='#4b5563')
    ax.set_title(title)
    ax.set_xlim(0, 255)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

def colorize_channel(arr_u8: np.ndarray, cmap: str = 'viridis') -> str:
    """Apply a colormap to a single-channel uint8 image and return base64 PNG.
    Uses OpenCV's applyColorMap for fast colorization.
    """
    cmap_map = {
        'viridis': cv2.COLORMAP_VIRIDIS,
        'turbo': cv2.COLORMAP_TURBO,
        'jet': cv2.COLORMAP_JET,
        'magma': cv2.COLORMAP_MAGMA,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
    }
    code = cmap_map.get(cmap, cv2.COLORMAP_VIRIDIS)
    color_bgr = cv2.applyColorMap(arr_u8, code)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return encode_image(color_rgb)

def process_image(image_file, mode: str = 'rgb'):
    """Process uploaded image and return requested channels by mode: rgb|hsv|yiq"""
    # Read image from uploaded file
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image file")
    
    # Convert BGR to RGB
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Precompute center
    center_y, center_x = img.shape[0]//2, img.shape[1]//2

    # Prepare base result
    result = {
        'original': encode_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        'dimensions': {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': img.shape[2]
        }
    }

    # NTSC YIQ conversion (use RGB normalized to [0,1] per NTSC procedure)
    trans_matrix = np.array([
        [0.299,  0.587,  0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523,  0.312]
    ], dtype=np.float32)
    rgb_norm = RGB.astype(np.float32) / 255.0
    im_yiq = np.dot(rgb_norm, trans_matrix.T)
    y_f = im_yiq[:, :, 0]
    i_f = im_yiq[:, :, 1]
    q_f = im_yiq[:, :, 2]

    def scale_fixed_range(a: np.ndarray, max_abs: float) -> np.ndarray:
        """Map values in [-max_abs, max_abs] to [0,255] for display."""
        a = np.clip(a, -max_abs, max_abs)
        scaled = (a + max_abs) * (255.0 / (2.0 * max_abs))
        return scaled.astype(np.uint8)

    # Display versions: Y scaled to [0,255]; I/Q mapped using NTSC fixed ranges
    # Standard NTSC ranges (approx): I in [-0.5957, 0.5957], Q in [-0.5226, 0.5226]
    y = np.uint8(np.clip(y_f * 255.0, 0, 255))
    i = scale_fixed_range(i_f, 0.5957)
    q = scale_fixed_range(q_f, 0.5226)
    
    # Populate selected mode
    mode = (mode or 'rgb').lower()

    if mode == 'rgb':
        dR = RGB[:,:,0]
        dG = RGB[:,:,1]
        dB = RGB[:,:,2]
        result['rgb'] = {
            'r_channel': encode_image(cv2.cvtColor(dR, cv2.COLOR_GRAY2RGB)),
            'g_channel': encode_image(cv2.cvtColor(dG, cv2.COLOR_GRAY2RGB)),
            'b_channel': encode_image(cv2.cvtColor(dB, cv2.COLOR_GRAY2RGB)),
            'sample_values': {
                'r': int(dR[center_y, center_x]),
                'g': int(dG[center_y, center_x]),
                'b': int(dB[center_y, center_x])
            }
        }
    elif mode == 'hsv':
        HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)
        dH = HSV[:,:,0]
        dS = HSV[:,:,1]
        dV = HSV[:,:,2]
        result['hsv'] = {
            'h_channel': encode_image(cv2.cvtColor(dH, cv2.COLOR_GRAY2RGB)),
            's_channel': encode_image(cv2.cvtColor(dS, cv2.COLOR_GRAY2RGB)),
            'v_channel': encode_image(cv2.cvtColor(dV, cv2.COLOR_GRAY2RGB)),
            'hsv_full': encode_image(cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)),
            'sample_values': {
                'h': int(dH[center_y, center_x]),
                's': int(dS[center_y, center_x]),
                'v': int(dV[center_y, center_x])
            }
        }
    elif mode == 'yiq':
        result['yiq'] = {
            'y_channel': colorize_channel(y, 'viridis'),
            'i_channel': colorize_channel(i, 'viridis'),
            'q_channel': colorize_channel(q, 'viridis'),
            'sample_values': {
                'y': float(y_f[center_y, center_x]),
                'i': float(i_f[center_y, center_x]),
                'q': float(q_f[center_y, center_x])
            },
            'histograms': {
                'y_hist': plot_histogram(y, 'Y channel histogram'),
                'i_hist': plot_histogram(i, 'I channel histogram'),
                'q_hist': plot_histogram(q, 'Q channel histogram')
            }
        }
    else:
        # default to rgb
        dR = RGB[:,:,0]
        dG = RGB[:,:,1]
        dB = RGB[:,:,2]
        result['rgb'] = {
            'r_channel': encode_image(cv2.cvtColor(dR, cv2.COLOR_GRAY2RGB)),
            'g_channel': encode_image(cv2.cvtColor(dG, cv2.COLOR_GRAY2RGB)),
            'b_channel': encode_image(cv2.cvtColor(dB, cv2.COLOR_GRAY2RGB)),
            'sample_values': {
                'r': int(dR[center_y, center_x]),
                'g': int(dG[center_y, center_x]),
                'b': int(dB[center_y, center_x])
            }
        }

    return result

@app.route('/')
def index():
    """Serve the HTML page"""
    return send_file('index.html')

@app.route('/convert', methods=['POST'])
def convert_image():
    """Handle image upload and processing"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        mode = request.form.get('mode', 'rgb')
        result = process_image(image_file, mode)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 Image RGB & HSV Converter Server")
    print("="*60)
    print("Server berjalan di: http://localhost:5000")
    print("Buka browser dan akses: http://localhost:5000")
    print("Tekan Ctrl+C untuk menghentikan server")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)