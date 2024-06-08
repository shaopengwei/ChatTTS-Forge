# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Package      : 
@FileName     : test
@Time         : 2024/6/7 15:56
@Author       : shaopengwei@hotmail.com
@License      : (C)Copyright 2024
@Version      : 1.0.0
@Desc         : None
"""

import ChatTTS
import torch
import torchaudio

if __name__ == '__main__':
    chat = ChatTTS.Chat()
    chat.load_models(source='local', local_path='D:\\work\\python\\ChatTTS\\models', compile=False)  # 设置为True以获得更快速度

    rand_spk = chat.sample_random_speaker()

    params_infer_code = {
        'spk_emb': rand_spk,  # add sampled speaker
        'temperature': .3,  # using custom temperature
        'top_P': 0.7,  # top P decode
        'top_K': 20,  # top K decode
    }

    ###################################
    # For sentence level manual control.

    # use oral_(0-9), laugh_(0-2), break_(0-7)
    # to generate special token in text to synthesize.
    params_refine_text = {
        'prompt': '[oral_2][laugh_1][break_6]'
    }

    texts = """
chat T T S 是一款强大的对话式文本转语音模型。它有中英混读和多说话人的能力。
chat T T S 不仅能够生成自然流畅的语音，还能控制[laugh]笑声啊[laugh]，
停顿啊[uv_break]语气词啊等副语言现象[uv_break]。这个韵律超越了许多开源模型[uv_break]。
请注意，chat T T S 的使用应遵守法律和伦理准则，避免滥用的安全风险。[uv_break]
""".replace('\n', '')
    wav = chat.infer(texts, params_refine_text=params_refine_text, params_infer_code=params_infer_code, use_decoder=False)

    torchaudio.save("output3.wav", torch.from_numpy(wav[0]), 24000, backend='soundfile')
