#!/usr/bin/env python3
"""本地 demo 前端（阶段 5）：连接 llama-server（OpenAI 兼容接口）的 Gradio 聊天页。

先启动 llama-server（GGUF 文件从 Colab 拷回本地后）：
    llama-server -m models/qwen2.5-1.5b-legal-q8_0.gguf --port 8080
再运行本脚本：
    python deploy/gradio_app.py
浏览器打开 http://127.0.0.1:7860
"""
import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

SYSTEM = (
    "你是一名专业的中国法律咨询助手。回答时：1）先给出结论；"
    "2）引用具体法条（格式如《民法典》第一千二百五十四条）；"
    "3）最后提醒本回答不构成正式法律意见。"
)


def reply(message, history):
    msgs = [{"role": "system", "content": SYSTEM}]
    msgs += [{"role": h["role"], "content": h["content"]} for h in history]
    msgs.append({"role": "user", "content": message})
    partial = ""
    stream = client.chat.completions.create(
        model="local", messages=msgs, max_tokens=512, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            partial += delta.content
            yield partial


demo = gr.ChatInterface(
    reply,
    type="messages",
    title="法律问答助手（Qwen2.5-1.5B 微调版）",
    description="模型输出仅供参考，不构成法律意见。",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
