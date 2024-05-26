import gradio as gr
import requests
import time

from googletrans import Translator
from googletrans import LANGUAGES as TRANSLATOR_LANGUAGES

from faster_whisper import WhisperModel
#docker run -p 8080:8080 -p 1337:1337 -p 7900:7900 --shm-size="5g" hlohaus789/g4f:latest
# googletrans==3.1.0a0

def translate_to_language(text, target_language="Russian"):
    translator = Translator()
    return translator.translate(text, src='auto', dest=target_language).text

def analyze_text(prompt):
    url = "http://localhost:1337/v1/chat/completions"
    body = {
        "model": "llama3-70b-instruct",
        "stream": False,
        'messages': [{'role': 'system',
                      'content': f'Теперь ты анализатор речи. Тебе нужно будет провести анализ речи. Выдели все слова-паразиты, собери их по группам и выдай в качестве ответа с примером к каждой группе. Твой ответ будет выглядеть следующим образом: \
                       *слово-паразит1*: пример предложения1, \
                       *слово-паразит2*: пример предложения2\
                       Тебе нужно будет выдать рекомендацию по улучшению речи, что изменить, какие слова лучше не использовать.\
                       Теперь жди сообщения пользователя. В ответ обязательно нужно что-то отдать, нельзя не ответить. Также отметь хорошие моменты в речи, которые пользователю лучше сохранить\
                       '},
                     {'role': 'user',
                      'content': f'{prompt}'
                      }]}
    
    json_response = requests.post(url, json=body).json().get('choices', [])
    while json_response == []:
        time.sleep(1)
        json_response = requests.post(url, json=body).json().get('choices', [])
        if json_response == []:
            continue
        else:
            break
    return json_response[0].get("message").get("content")


def pipeline(input_audio, input_text, language="russian"):
    print("inputs", input_audio, input_text)
    # faster whisper model declaration
    model = WhisperModel("medium", device="cuda", compute_type="float16")
    segments, info = model.transcribe(input_audio, beam_size=5)

    # variables for working with word_timestamp mode 
    text_arr = []
    last_segment_end = 0.0

    for segment in segments:
        text_arr.append(segment.text)
        if segment.start - last_segment_end > 5.0:
            print("There was a delay of more than 5 seconds between phrases")
        if segment.start - last_segment_end  > 10.0:
            print("There was a delay of more than 10 seconds between phrases")
        print("[%.1fs -> %.1fs] %s" % (segment.start, segment.end, segment.text))
        last_segment_end = segment.end


    result = "".join(text_arr)
    analyzed_speech = analyze_text(result)

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

            gr.Textbox(label="Дополнительные настройки для модели", info="Сюда можно написать специализированную лексику, имена. Также можно включить пожелания при выдаче рекомендации"
            ),

            gr.Dropdown(label="Язык рекомендаций", info="Вывод моделей будет на выбранном языке", choices=TRANSLATOR_LANGUAGES.values(), allow_custom_value=True)
            
        ]

    outputs = gr.Markdown("Тестовый текст *Markdown*")

    btn = gr.Button("Анализ речи")
    btn.click(fn=pipeline, inputs=inputs, outputs=outputs)
    demo.launch()

