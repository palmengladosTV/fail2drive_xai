import cv2
import os
import sys
import re
from tqdm import tqdm


def extract_number(filename):
    """Sucht die erste Zahl in einem Dateinamen und gibt sie als Integer zurück."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1


def put_text_with_shadow(img, text, position):
    """Hilfsfunktion für gut lesbaren Text auf Bildern."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Schwarzer Schatten
    cv2.putText(img, text, (position[0] + 2, position[1] + 2),
                font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Weißer Text
    cv2.putText(img, text, position,
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def create_combined_videos(master_folder, categories_root_folder, fps=2):
    # 1. MASTER-Bilder einlesen und nach ihrer Nummer indexieren
    print("Scanne Master-Ordner...")
    master_images = [img for img in os.listdir(master_folder) if img.endswith(".png")]
    if not master_images:
        print("Fehler: Keine Master-Bilder gefunden.")
        return

    master_dict = {extract_number(img): os.path.join(master_folder, img) for img in master_images}
    master_keys = sorted(master_dict.keys())

    # 2. KATEGORIEN-Bilder einlesen und gruppieren
    print("Scanne Kategorien-Ordner...")
    subfolders = [f for f in os.listdir(categories_root_folder) if
                  os.path.isdir(os.path.join(categories_root_folder, f))]

    category_dict = {}

    for subfolder in subfolders:
        subfolder_path = os.path.join(categories_root_folder, subfolder)
        images = [img for img in os.listdir(subfolder_path) if img.endswith(".png")]

        for image_name in images:
            num = extract_number(image_name)
            name_without_ext = image_name.rsplit('.', 1)[0]
            parts = name_without_ext.rsplit('_', 1)

            if len(parts) == 2:
                category_name = parts[0]
                if category_name not in category_dict:
                    category_dict[category_name] = {}

                image_path = os.path.join(subfolder_path, image_name)
                category_dict[category_name][num] = (image_path, image_name)

    if not category_dict:
        print("Es wurden keine passenden Kategorien-Bilder gefunden.")
        return

    # 3. Dimensionen für das zusammengefügte Bild bestimmen
    first_master_img = cv2.imread(master_dict[master_keys[0]])
    master_h, master_w, _ = first_master_img.shape

    cat_h, cat_w = master_h, master_w
    for cat_data in category_dict.values():
        if cat_data:
            first_cat_path = list(cat_data.values())[0][0]
            temp_img = cv2.imread(first_cat_path)
            if temp_img is not None:
                cat_h, cat_w, _ = temp_img.shape
            break

    new_cat_w = int(cat_w * (master_h / cat_h))

    # 4. VIDEOS ERSTELLEN
    for category_name, cat_images_by_num in category_dict.items():
        output_video_name = f"{category_name}_Combined.mp4"
        output_video_path = os.path.join(categories_root_folder, output_video_name)

        final_width = new_cat_w + master_w

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (final_width, master_h))

        print(f"\nErstelle kombiniertes Video für: {category_name}")

        for num in tqdm(master_keys, desc=f"Rendere {category_name}", unit="Frame"):

            # HIER IST DIE ÄNDERUNG:
            # Wenn es zu dieser Nummer kein Kategorienbild gibt, überspringen wir den Frame komplett.
            if num not in cat_images_by_num:
                continue

            # 4.1 Master-Bild laden
            master_img = cv2.imread(master_dict[num])
            master_filename = os.path.basename(master_dict[num])
            put_text_with_shadow(master_img, master_filename, (30, 50))

            # 4.2 Kategorien-Bild laden
            cat_path, cat_filename = cat_images_by_num[num]
            cat_img = cv2.imread(cat_path)

            # Größe anpassen
            if cat_img.shape[:2] != (master_h, new_cat_w):
                cat_img = cv2.resize(cat_img, (new_cat_w, master_h))

            put_text_with_shadow(cat_img, cat_filename, (30, 50))

            # 4.3 Bilder horizontal zusammenfügen (Links Kategorie, Rechts Master)
            combined_frame = cv2.hconcat([cat_img, master_img])

            # 4.4 Frame ins Video schreiben
            video.write(combined_frame)

        video.release()

    print("\nAlle Master-Kategorien-Videos wurden erfolgreich erstellt!")


if __name__ == "__main__":
    print("--- Video Kombinierer ---")

    if len(sys.argv) == 3:
        master_dir = sys.argv[1]
        cat_dir = sys.argv[2]
    else:
        master_dir = input("1. Bitte gib den Pfad zum Ordner mit den MASTER-Bildern ein: ")
        cat_dir = input("2. Bitte gib den Pfad zum Hauptordner der KATEGORIEN ein: ")

    if not os.path.isdir(master_dir):
        print(f"Fehler: Master-Ordner '{master_dir}' nicht gefunden.")
    elif not os.path.isdir(cat_dir):
        print(f"Fehler: Kategorien-Ordner '{cat_dir}' nicht gefunden.")
    else:
        create_combined_videos(master_dir, cat_dir)