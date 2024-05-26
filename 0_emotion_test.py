from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import torch
import gradio as gr
import requests
import time

from googletrans import Translator
from googletrans import LANGUAGES as TRANSLATOR_LANGUAGES

from transformers import pipeline
from transformers.utils import is_flash_attn_2_available


def translate_to_language(text, target_language="Russian"):
    translator = Translator()
    return translator.translate(text, src='auto', dest=target_language).text

def analyze_text(prompt, pauses="", speech_emotions=""):
    url = "http://localhost:1337/v1/chat/completions"
    body = {
        "model": "llama3-70b-instruct",
        #"model": "gpt-3.5-turbo",
        "stream": False,
        'messages': [{'role': 'system',
                      'content': f'Теперь ты анализатор речи. Тебе нужно будет провести анализ речи. Выдели все слова-паразиты, собери их по группам и выдай в качестве ответа с примером к каждой группе. Твой ответ будет выглядеть следующим образом: \
                       *слово-паразит1*: пример предложения1, \
                       *слово-паразит2*: пример предложения2\
                       Тебе нужно будет выдать рекомендацию по улучшению речи, что изменить, какие слова лучше не использовать.\
                       Также тебе нужно выдать рекомендацию по уменьшению длительности пауз, если такие есть. \
                       Теперь жди сообщения пользователя. В ответ обязательно нужно что-то отдать, нельзя не ответить. Также отметь хорошие моменты в речи, которые пользователю лучше сохранить\
                       '},
                     {'role': 'user',
                      'content': f'{prompt} \
                      \n Примеры предложений с паузами более 5 секунд: {pauses} \
                      \n Текст был сказан со следующими эмоциями {speech_emotions}, чем большее преобладание эмоции в речи тем ближе значение к 1.00 \
                      '
                      }]}
    
    json_response = requests.post(url, json=body).json().get('choices', [])
    if json_response == []:
        time.sleep(10)
        json_response = requests.post(url, json=body).json().get('choices', [])
    return json_response[0].get("message").get("content")

def define_pauses(chunks, trashhold=5.0):
    last_segment_end = 0.0
    above_trashhold_pause_phrases = []
    for chunk in chunks:
        if chunk["timestamp"][0] - last_segment_end > trashhold:
            print("There was a delay of 5 seconds between last phrase and")
            print(chunk["text"])
            above_trashhold_pause_phrases.append(chunk["text"])
        last_segment_end = chunk["timestamp"][1]

    result = "".join(above_trashhold_pause_phrases)
    return result

def emotion_recognition (input_audio):
    pipe = pipeline("audio-classification", device="cuda", model="DunnBC22/wav2vec2-base-Speech_Emotion_Recognition" )
    output = pipe(input_audio)
    return output

def pipeline_stt(input_audio, input_text, language="russian"):
    print("inputs", input_audio, input_text)

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3", 
        torch_dtype=torch.float16,
        device="cuda:0", 
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    outputs = pipe(
        input_audio,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )

    result = outputs["text"]
    timestamps = outputs["chunks"]
    analyzed_speech = analyze_text(result, define_pauses(timestamps), emotion_recognition(input_audio))
    print(result)

    return str(analyzed_speech)


with gr.Blocks() as demo:
    gr.Markdown("Прототип приложения для анализа речи при подготовке к публичным выступлениям")
    with gr.Row():
        inputs = [
            gr.Audio(
                sources=['upload', 'microphone'],
                format='mp3',
                type='filepath'
            ),
            gr.Textbox(label="Дополнительные настройки для модели", info="Сюда можно написать специализированную лексику, имена. \
                        Также можно включить пожелания при выдаче рекомендации"
            ),
            gr.Dropdown(label="Язык рекомендаций", info="Вывод моделей будет на выбранном языке",
                         choices=TRANSLATOR_LANGUAGES.values(), allow_custom_value=True)
        ]

    outputs = gr.Markdown("Загрузите файл в любом аудиоформате или запишите свою речь, \
                        укажите язык, на котором хотите получить рекомендацию. Нажмите кнопку *Анализ речи*")

    btn = gr.Button("Анализ речи")
    btn.click(fn=pipeline_stt, inputs=inputs, outputs=outputs)
    demo.launch(share=False)

