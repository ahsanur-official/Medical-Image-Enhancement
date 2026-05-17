from flask import Flask, request, render_template, jsonify
import os
import tempfile
import base64
import io
import cv2
import numpy as np
from medical_image_project import MedicalImageEnhancer
import traceback

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def api_process():
    # Expect 'image' file and 'technique' field
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    technique = request.form.get('technique', 'comprehensive')
    # Optional parameters
    clahe_clip = float(request.form.get('clahe_clip') or 2.0)
    gamma = float(request.form.get('gamma') or 1.0)
    # kernel sizes
    try:
        kernel = int(request.form.get('kernel') or 5)
    except Exception:
        kernel = 5
    # ensure odd and >=1
    if kernel < 1:
        kernel = 1
    if kernel % 2 == 0:
        kernel += 1

    try:
        sobel_ksize = int(request.form.get('sobel_ksize') or 3)
    except Exception:
        sobel_ksize = 3
    if sobel_ksize not in (1, 3, 5, 7):
        sobel_ksize = 3

    # Save uploaded file to temporary location
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(img_file.filename)[1])
    os.close(fd)
    img_file.save(path)

    try:
        enhancer = MedicalImageEnhancer(path)

        # Support histogram matching when 'template' is provided
        if 'template' in request.files and request.files['template'].filename != '' and technique in ['match','hist_match']:
            tpl_file = request.files['template']
            fd2, tpl_path = tempfile.mkstemp(suffix=os.path.splitext(tpl_file.filename)[1])
            os.close(fd2)
            tpl_file.save(tpl_path)
            try:
                tpl_enh = MedicalImageEnhancer(tpl_path)
                out = enhancer.histogram_matching(enhancer.original, tpl_enh.original)
            finally:
                try: os.remove(tpl_path)
                except Exception: pass

        else:
            if technique == 'comprehensive':
                out = enhancer.comprehensive_enhancement()
            elif technique == 'advanced':
                out = enhancer.advanced_enhancement()
            elif technique == 'clahe':
                out = enhancer.clahe(clip_limit=clahe_clip)
            elif technique == 'average':
                out = enhancer.average_filter(kernel_size=(kernel,kernel))
            elif technique == 'median':
                out = enhancer.median_filter(kernel_size=kernel)
            elif technique == 'sobel':
                out = enhancer.sobel_edge_detection(ksize=sobel_ksize)
            elif technique == 'laplacian':
                out = enhancer.laplacian_enhancement()
            elif technique == 'histogram':
                out = enhancer.histogram_equalization()
            elif technique == 'gamma':
                # gamma >1 brightens; <1 darkens
                out = enhancer.adaptive_gamma_correction(gamma=gamma)
            else:
                out = enhancer.comprehensive_enhancement()

        # Compute histogram features for both original and processed
        orig_features, orig_hist = enhancer.compute_histogram_features(enhancer.original)
        proc_features, proc_hist = enhancer.compute_histogram_features(out)

        # Ensure all values are native Python types so JSON serialization won't fail
        def sanitize_features(d):
            return {k: (int(v) if isinstance(v, (np.integer,)) else float(v)) for k, v in d.items()}

        def sanitize_hist(h):
            return [float(x) for x in h]

        orig_features_s = sanitize_features(orig_features)
        proc_features_s = sanitize_features(proc_features)
        orig_hist_s = sanitize_hist(orig_hist)
        proc_hist_s = sanitize_hist(proc_hist)

        # Encode processed image as PNG base64
        success, encoded = cv2.imencode('.png', out)
        if not success:
            raise RuntimeError('Failed to encode image')
        b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

        response = {
            'image_name': img_file.filename,
            'technique': technique,
            'processed_image_png_base64': b64,
            'original_features': orig_features_s,
            'processed_features': proc_features_s,
            'original_histogram': orig_hist_s,
            'processed_histogram': proc_hist_s
        }

        return jsonify(response)

    except Exception as e:
        # Log full traceback to console for debugging
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        try:
            os.remove(path)
        except Exception:
            pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
