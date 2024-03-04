import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import concurrent.futures

class ParaphraseModel:

    def __init__(self, model_ckpt="ramsrigouthamg/t5-large-paraphraser-diverse-high-quality", beam_nums=5) -> None:

        self.beam_nums = beam_nums
        self.model_ckpt = model_ckpt
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_ckpt)
        self.pipe = pipeline(task="text2text-generation", model=self.model, tokenizer=self.tokenizer)


    def paraphrase(self, input_text: str) -> str:

        def process_sentence(self, sentence):
            return self.pipe(sentence, max_length=60, num_beams=self.beam_nums,
                             num_return_sequences=1, temperature=1.5)[0]['generated_text'].split(": ", 1)[1]

        sentences = sent_tokenize(input_text)
        paraphrased_text = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(lambda s: process_sentence(self, s), sentences)

        paraphrased_text = " ".join(results)
        return paraphrased_text
        