from PIL import Image
import os
import json

def crop_center(input_path, output_path, crop_width, crop_height):
    """
    Schneidet den mittleren Bereich eines Bildes auf eine bestimmte Größe zu.
    """
    try:
        with Image.open(input_path) as img:
            original_width, original_height = img.size
            left = (original_width - crop_width) / 2
            top = (original_height - crop_height) / 2
            right = (original_width + crop_width) / 2
            bottom = (original_height + crop_height) / 2
            if left < 0 or top < 0 or right > original_width or bottom > original_height:
                print("Fehler: Der gewünschte Ausschnitt ist größer als das Originalbild.")
                return
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(output_path)
            print(f"Bild erfolgreich in der Mitte zugeschnitten und gespeichert unter: {output_path}")
    except FileNotFoundError:
        print(f"Fehler: Eingabebild nicht gefunden unter {input_path}")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

def get_image_dimensions(image_path):
    """
    Gibt die Abmessungen (Breite, Höhe) und die Dateigröße eines Bildes aus.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"\n--- Bildinformationen für: {image_path} ---")
            print(f"Dimensionen: {width}px Breite x {height}px Höhe")
            file_size = os.path.getsize(image_path)
            print(f"Dateigröße: {file_size / 1024:.2f} KB")
            return width, height
    except FileNotFoundError:
        print(f"Fehler: Bild '{image_path}' nicht gefunden.")
        return None, None
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None, None

def crop_from_json(input_path, output_path, json_string):
    """
    Schneidet ein Bild basierend auf einem JSON-String mit Koordinaten zu.
    """
    try:
        coords = json.loads(json_string)
        left = coords['x1']
        top = coords['y1']
        right = coords['x2']
        bottom = coords['y2']
        with Image.open(input_path) as img:
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(output_path)
            print(f"Bild erfolgreich via JSON zugeschnitten und gespeichert unter: {output_path}")
    except (json.JSONDecodeError, KeyError):
        print("Fehler: JSON muss 'x1', 'y1', 'x2', 'y2' enthalten.")
    except FileNotFoundError:
        print(f"Fehler: Eingabebild nicht gefunden unter {input_path}")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")


if __name__ == "__main__":
    # Dieses Skript ist jetzt der ZWEITE Schritt im Workflow.
    # Es nimmt das Bild, auf das bereits Text gezeichnet wurde, und schneidet es zu.
    
    input_img_with_text = "temp_text_added.png"
    final_output_img = "final_high_quality.png"

    # Holen Sie die Dimensionen des Bildes mit Text
    width_with_text, height_with_text = get_image_dimensions(input_img_with_text)

    if width_with_text is not None:
        # Schneiden Sie es auf 50% seiner Größe zu, um den mittleren Teil zu erhalten
        desired_width = width_with_text // 2
        desired_height = height_with_text // 2
        
        print(f"\n--- Schneide das Bild mit Text auf {desired_width}x{desired_height}px zu ---")
        crop_center(input_img_with_text, final_output_img, desired_width, desired_height)

        # Zeigen Sie die Dimensionen des finalen Bildes an
        get_image_dimensions(final_output_img)
