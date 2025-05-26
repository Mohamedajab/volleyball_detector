import easyocr
import cv2

def detect_jersey_numbers(players, min_conf=0.3):
    reader = easyocr.Reader(['en'], gpu=False)

    for player in players:
        try:
            gray = cv2.cvtColor(player.image, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(gray)

            if results:
                top = max(results, key=lambda x: x[2])
                text = top[1]
                conf = top[2]

                if conf >= min_conf and text.isdigit():
                    player.jersey_number = text
                    player.ocr_confidence = conf
                    player.needs_review = False
                else:
                    player.ocr_confidence = conf
                    player.needs_review = True
            else:
                player.ocr_confidence = 0.0
                player.needs_review = True

        except Exception as e:
            print("OCR Error:", e)
            player.ocr_confidence = 0.0
            player.needs_review = True

    return players
