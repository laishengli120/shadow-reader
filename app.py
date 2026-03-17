from flask import Flask, request, jsonify
from gtts import gTTS
from pydub import AudioSegment
import io
import base64

app = Flask(__name__)

ACCENT_MAP = {
    'US': ('en', 'com'),
    'UK': ('en', 'co.uk'),
    'AU': ('en', 'com.au')
}

@app.route('/')
def index():
    # 记得将前端 HTML 放入 templates/index.html 中
    from flask import render_template
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_audio():
    try:
        data = request.json
        text = data.get('text', '')
        interval = float(data.get('interval', 2.0))
        accent_code = data.get('accent', 'US')
        
        lang, tld = ACCENT_MAP.get(accent_code, ('en', 'com'))

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return jsonify({"error": "文本为空"}), 400

        combined_audio = AudioSegment.empty()
        silence_segment = AudioSegment.silent(duration=int(interval * 1000))
        
        timings = []
        current_time_ms = 0

        for i, line in enumerate(lines):
            # 生成单句音频
            tts = gTTS(text=line, lang=lang, tld=tld)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            line_audio = AudioSegment.from_file(mp3_fp, format="mp3")
            line_duration = len(line_audio)
            
            # 记录当前句子的时间戳数据 (用于前端高亮)
            timings.append({
                "index": i,
                "text": line,
                "start_time": current_time_ms / 1000.0,  # 转换为秒
                "end_time": (current_time_ms + line_duration) / 1000.0
            })
            
            combined_audio += line_audio
            current_time_ms += line_duration
            
            # 插入静音
            if i < len(lines) - 1:
                combined_audio += silence_segment
                current_time_ms += len(silence_segment)

        # 导出完整音频并转为 Base64
        out_fp = io.BytesIO()
        combined_audio.export(out_fp, format="mp3")
        base64_audio = base64.b64encode(out_fp.getvalue()).decode('utf-8')

        return jsonify({
            "audio_base64": base64_audio,
            "timings": timings
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "生成音频时发生错误"}), 500

if __name__ == '__main__':
    app.run(debug=True)