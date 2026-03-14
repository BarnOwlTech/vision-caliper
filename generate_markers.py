import cv2
import cv2.aruco as aruco
import numpy as np


def generate_aruco_markers():
    """
    Генерирует 4 ArUco маркера для калибровки (OpenCV 4.7.0+)
    """
    # Новый API для OpenCV 4.7.0+
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    marker_size = 300  # пикселей
    border_size = 50  # белая рамка вокруг
    total_size = marker_size + 2 * border_size

    # ИСПРАВЛЕНИЕ: Создаем 2D изображение (одноканальное), а не 3D
    output_image = np.ones((total_size * 2, total_size * 2), dtype=np.uint8) * 255

    marker_ids = [0, 1, 2, 3]
    labels = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]

    for i, marker_id in enumerate(marker_ids):
        row = i // 2
        col = i % 2
        x = col * total_size
        y = row * total_size

        # Генерируем маркер (возвращает 2D массив)
        marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

        # ИСПРАВЛЕНИЕ: Теперь размерности совпадают (2D в 2D)
        output_image[y + border_size:y + border_size + marker_size,
        x + border_size:x + border_size + marker_size] = marker_image

        # Добавляем подпись (для cv2.putText нужно 3D изображение, поэтому конвертируем)
        output_color = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(output_color, f"ID:{marker_id}", (x + 10, y + border_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        cv2.putText(output_color, labels[i], (x + 10, y + total_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        output_image = cv2.cvtColor(output_color, cv2.COLOR_BGR2GRAY)

    # Конвертируем в BGR для финальных надписей
    output_final = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

    # Добавляем общую информацию
    cv2.putText(output_final, "ARUCO MARKERS FOR CALIBRATION", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(output_final, "Print this page at 100% scale (no scaling!)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(output_final, "Marker size: 30mm x 30mm (measure after printing)", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Сохраняем
    cv2.imwrite("aruco_markers.png", output_final)
    print("✅ Маркеры сохранены в 'aruco_markers.png'")
    print("📏 Распечатайте на A4 без масштабирования (100%)")
    print("📐 Измерьте линейкой реальный размер маркера после печати")
    print(f"📋 Версия OpenCV: {cv2.__version__}")

    return output_final


if __name__ == "__main__":
    generate_aruco_markers()