from typing import Any
import googletrans
import translate

class Translator:
    def __init__(self, from_lang="vi", to_lang="en", method="google"):
        self.__method = method
        self.__from_lang = from_lang
        self.__to_lang = to_lang
        if method in "googletrans":
            self.translator = googletrans.Translator()
        elif method in "translate":
            self.translator = translate.Translator(from_lang=from_lang, to_lang=to_lang)

    def text_normalize(self, text: str):
        return text.lower()

    def __call__(self, text: str):
        text = self.text_normalize(text)
        return (
            self.translator.translate(text)
            if self.__method in "translate"
            else self.translator.translate(text, dest=self.__to_lang).text
        )
    


if __name__ == "__main__":
    translator = Translator()
    print(translator("con chó vừa béo vừa kiêu"))
