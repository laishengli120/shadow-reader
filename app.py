from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import edge_tts
import asyncio
import nest_asyncio
import io
import base64
import os
import tempfile

# 允许在 Flask 的同步线程中运行异步循环
nest_asyncio.apply()

app = Flask(__name__)

# 替换为微软 Edge Neural 的高质量真人发音人
ACCENT_MAP = {
    'US': 'en-US-AriaNeural',    # 美音女声 (非常自然)
    'UK': 'en-GB-SoniaNeural',   # 英音女声
    'AU': 'en-AU-NatashaNeural'  # 澳音女声
}

async def generate_single_line_audio(text, voice, output_path):
    """异步调用 edge-tts 生成单行音频并保存到临时文件"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_audio():
    try:
        data = request.json
        text = data.get('text', '')
        interval = float(data.get('interval', 2.0))
        accent_code = data.get('accent', 'US')
        
        voice = ACCENT_MAP.get(accent_code, 'en-US-AriaNeural')

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return jsonify({"error": "文本为空"}), 400

        combined_audio = AudioSegment.empty()
        silence_segment = AudioSegment.silent(duration=int(interval * 1000))
        
        timings = []
        current_time_ms = 0

        # 创建一个临时目录来存放生成的单句音频
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i, line in enumerate(lines):
                tmp_file_path = os.path.join(tmpdirname, f"line_{i}.mp3")
                
                # 运行异步任务生成音频
                asyncio.run(generate_single_line_audio(line, voice, tmp_file_path))
                
                # 使用 pydub 读取刚刚生成的音频
                line_audio = AudioSegment.from_file(tmp_file_path, format="mp3")
                line_duration = len(line_audio)
                
                timings.append({
                    "index": i,
                    "text": line,
                    "start_time": current_time_ms / 1000.0,
                    "end_time": (current_time_ms + line_duration) / 1000.0
                })
                
                combined_audio += line_audio
                current_time_ms += line_duration
                
                if i < len(lines) - 1:
                    combined_audio += silence_segment
                    current_time_ms += len(silence_segment)

        # 导出并转为 Base64
        out_fp = io.BytesIO()
        combined_audio.export(out_fp, format="mp3")
        base64_audio = base64.b64encode(out_fp.getvalue()).decode('utf-8')

        return jsonify({
            "audio_base64": base64_audio,
            "timings": timings
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)