from rapidfuzz import fuzz
import re

def get_ocr_results(query, ocr_results):
    def check(text):
        score = fuzz.token_set_ratio(text.lower(), query.lower())
        print(text, score)
        if score >= 80:
            return True
        return False

    for ocr_text in ocr_results: 
        if check(ocr_text) == True:
            return True
        
    return False