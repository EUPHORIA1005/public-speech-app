import whisper
import time

#model = whisper.load_model("medium")
#result = model.transcribe("samples/test.mp3", word_timestamps=False)
#print(result["text"])

chunks = [{''}]

last_segment_end = 0.0
above_trashhold_pause_phrases = []
for chunk in chunks:
    print(chunk)
    if chunk["timestamp"][0] - last_segment_end > 122.0:
        #print("There was a delay of 12 seconds between last phrase and")
        above_trashhold_pause_phrases.append(chunk["text"])
    last_segment_end = chunk["timestamp"][1]


print("".join(above_trashhold_pause_phrases))




# start_time = time.time()

# print("--- %s seconds ---" % (time.time() - start_time))

# def test(model_type):
#     start_time = time.time()
#     model = whisper.load_model(model_type)
#     result = model.transcribe("test_standup.mp3")
#     total_time=time.time() - start_time
#     print("Total time with model " + str(model_type) + " " + str(total_time))
#     print(result["text"])

#     print("--- %s seconds ---" % (time.time() - start_time))

# test("medium")
# test("large")



