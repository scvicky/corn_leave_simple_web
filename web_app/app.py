from flask import Flask, render_template, request, jsonify
import os
import traceback
from predict import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 保存上传的文件
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        print(f"✅ 文件已保存: {filepath}")
        
        # 进行预测
        detections = predict(filepath)
        
        print(f"✅ 检测完成，结果: {detections}")
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'detections': detections
        })
    
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
