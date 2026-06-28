# Shadow Reader

Shadow Reader 是一个基于 Flask 的文本朗读和语音生成 Web 应用。项目把多种 TTS 服务商封装成统一接口，前端提供文本输入、语音选择和音频生成体验，后端负责调用语音服务、处理音频并返回结果。

## 功能

- 浏览器端输入文本并生成朗读音频
- 支持多种 TTS Provider 的统一接入
- 支持 Edge TTS、gTTS、系统离线语音、DashScope 等语音来源
- 使用 OpenAI SDK 兼容接口接入可配置模型服务
- 使用 `pydub` 处理音频片段
- 通过测试覆盖音频流水线的关键逻辑

## 技术栈

- Python / Flask
- Flask-CORS
- Flask-Limiter
- OpenAI Python SDK
- pydub
- edge-tts、gTTS、pyttsx3、dashscope
- pytest

## 环境准备

建议使用 Python 3.11 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pydub` 需要系统安装 FFmpeg：

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

## 运行

```bash
python app.py
```

默认打开本地 Flask 服务后，在浏览器访问终端提示的地址。

## 配置

项目会读取环境变量中的日志等级、API Key、服务地址和模型参数。不同 TTS Provider 可能需要不同的密钥或本机语音环境。请不要把真实密钥提交到仓库。

## 测试

```bash
pytest
```

当前测试重点在音频生成和处理流水线。新增 Provider 时建议补充失败重试、超时、音频格式和空文本输入的测试。

## 项目结构

```text
app.py                         # Flask 应用、Provider 抽象和主要接口
templates/index.html           # 前端页面
requirements.txt               # Python 依赖
tests/test_audio_pipeline.py   # 音频流水线测试
```
