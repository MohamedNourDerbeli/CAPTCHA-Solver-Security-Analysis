from captcha.image import ImageCaptcha
import random
import string

def generate_captcha():
    image = ImageCaptcha()
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    image.write(captcha_text, f"captcha_{captcha_text}.png")
    return captcha_text

# Generate 100 CAPTCHAs
captchas = [generate_captcha() for _ in range(100)]
