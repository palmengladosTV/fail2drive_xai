import cv2
import os
import sys
from tqdm import tqdm


def create_category_videos(root_folder, fps=10):
    # 1. Alle Unterordner im Hauptverzeichnis finden
    # Wir filtern nur echte Ordner heraus
    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    if not subfolders:
        print(f"Fehler: Keine Unterordner im Verzeichnis '{root_folder}' gefunden.")
        return

    # Unterordner aufsteigend sortieren (01, 02, 03 ...)
    subfolders.sort()

    # 2. Bilder nach Kategorien gruppieren
    # Format: { "KategorieA": [ (pfad_zu_bild1, dateiname1), (pfad_zu_bild2, dateiname2) ], ... }
    category_dict = {}

    print("Scanne Ordnerstruktur und gruppiere Bilder nach Kategorien...")
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder)
        images = [img for img in os.listdir(subfolder_path) if img.endswith(".png")]

        # Da wir die Unterordner nacheinander (01, 02...) abarbeiten,
        # werden die Bilder chronologisch an die Kategorien-Listen angehängt.
        for image_name in images:
            # ".png" vom Namen abschneiden (z.B. "A_01.png" -> "A_01")
            name_without_ext = image_name.rsplit('.', 1)[0]

            # Am letzten Unterstrich trennen (z.B. "Meine_Kategorie_01" -> ["Meine_Kategorie", "01"])
            parts = name_without_ext.rsplit('_', 1)

            if len(parts) == 2:
                category_name = parts[0]

                # Wenn die Kategorie noch nicht existiert, neue Liste anlegen
                if category_name not in category_dict:
                    category_dict[category_name] = []

                # Bildpfad und Name für später speichern
                image_path = os.path.join(subfolder_path, image_name)
                category_dict[category_name].append((image_path, image_name))

    if not category_dict:
        print("Es konnten keine Bilder gefunden werden, die dem Namensschema (Kategorie_Nummer.png) entsprechen.")
        return

    print(f"\nEs wurden {len(category_dict)} verschiedene Kategorien gefunden:")
    for cat in category_dict.keys():
        print(f" - {cat} ({len(category_dict[cat])} Bilder)")
    print("-" * 40)

    # 3. Für jede Kategorie ein Video erstellen
    for category_name, image_data_list in category_dict.items():
        # Video wird im Hauptverzeichnis gespeichert
        output_video_name = f"{category_name}.mp4"
        output_video_path = os.path.join(root_folder, output_video_name)

        # Erstes Bild einlesen, um die Auflösung für das Video zu bestimmen
        first_image_path, _ = image_data_list[0]
        frame = cv2.imread(first_image_path)

        # Sicherheitscheck, falls ein Bild defekt ist
        if frame is None:
            print(f"Warnung: Konnte Bild {first_image_path} nicht lesen. Überspringe Kategorie '{category_name}'.")
            continue

        height, width, layers = frame.shape

        # VideoWriter initialisieren (MP4-Format)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        print(f"\nErstelle Video für Kategorie: {category_name} -> {output_video_name}")

        # Schrift-Einstellungen für den Dateinamen im Video
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (30, 50)
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2

        # Einzelne Bilder ins Video schreiben (mit eigenem Progressbar pro Kategorie)
        for img_path, img_name in tqdm(image_data_list, desc=f"Video '{category_name}'", unit="Frame"):
            frame = cv2.imread(img_path)

            if frame is None:
                continue

            # Schwarzen Schatten/Rand hinzufügen
            cv2.putText(frame, img_name, (position[0] + 2, position[1] + 2),
                        font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)

            # Den eigentlichen weißen Text schreiben
            cv2.putText(frame, img_name, position,
                        font, font_scale, color, thickness, cv2.LINE_AA)

            # Frame in das Video schreiben
            video.write(frame)

        # Video abschließen
        video.release()

    print("\nAlle Videos wurden erfolgreich erstellt und im Hauptverzeichnis gespeichert!")


if __name__ == "__main__":
    # Abfrage des Hauptordners
    if len(sys.argv) > 1:
        ordner = sys.argv[1]
    else:
        ordner = input("Bitte gib den Pfad zum Hauptordner ein (in dem die nummerierten Unterordner liegen): ")

    if not os.path.isdir(ordner):
        print(f"Der Ordner '{ordner}' existiert nicht. Bitte überprüfe den Pfad.")
    else:
        create_category_videos(ordner)