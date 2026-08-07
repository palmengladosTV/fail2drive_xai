import cv2
import os
import sys
from tqdm import tqdm


def create_video_from_images(folder_path, output_video_name="scenario.mp4", fps=30):
    # Alle PNG-Bilder im Ordner finden
    images = [img for img in os.listdir(folder_path) if img.endswith(".png")]

    if not images:
        print(f"Fehler: Keine PNG-Bilder im Ordner '{folder_path}' gefunden.")
        return

    # Bilder aufsteigend sortieren
    images.sort()

    # NEU: Den vollständigen Pfad für das Ausgabe-Video erstellen
    # So landet das Video im selben Ordner wie die Bilder
    output_video_path = os.path.join(folder_path, output_video_name)

    # Erstes Bild einlesen, um die Auflösung für das Video zu bestimmen
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # VideoWriter initialisieren (MP4-Format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Starte Videoerstellung...")
    print(f"Zielpfad: {output_video_path}")
    print(f"Frames: {len(images)} | FPS: {fps}")

    # Schrift-Einstellungen für den Dateinamen
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (30, 50)
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2

    for image_name in tqdm(images, desc="Verarbeite Bilder", unit="Frame"):
        img_path = os.path.join(folder_path, image_name)
        frame = cv2.imread(img_path)

        # Schwarzen Schatten/Rand hinzufügen
        cv2.putText(frame, image_name, (position[0] + 2, position[1] + 2),
                    font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)

        # Den eigentlichen weißen Text schreiben
        cv2.putText(frame, image_name, position,
                    font, font_scale, color, thickness, cv2.LINE_AA)

        # Frame in das Video schreiben
        video.write(frame)

    # Ressourcen freigeben
    video.release()
    print(f"\nFertig! Das Video wurde erfolgreich gespeichert.")


if __name__ == "__main__":
    # Abfrage des Ordnernamens über die Kommandozeile oder per Texteingabe
    if len(sys.argv) > 1:
        ordner = sys.argv[1]
    else:
        ordner = input("Bitte gib den Pfad zum Bilderordner ein: ")

    # Falls der Ordner nicht existiert
    if not os.path.isdir(ordner):
        print(f"Der Ordner '{ordner}' existiert nicht. Bitte überprüfe den Pfad.")
    else:
        create_video_from_images(ordner)