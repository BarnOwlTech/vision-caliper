import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import datetime

# =============================================================================
# КОНСТАНТЫ
# =============================================================================
ARUCO_MARKER_SIZE_MM = 23.0  # Размер маркера в мм (измерьте после печати!)

# =============================================================================
# КАЛИБРОВКА С ARUCO МАРКЕРАМИ
# =============================================================================
def calibrate_with_aruco(image):
    """
    Находит ArUco маркеры на изображении и вычисляет:
    1. Матрицу перспективы (для коррекции)
    2. Масштаб (пиксели/мм)

    Возвращает: (success, message, calibration_data)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Новый API для OpenCV 4.7.0+
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector_parameters = aruco.DetectorParameters()

    # Детектирование маркеров (новый API)
    detector = aruco.ArucoDetector(aruco_dict, detector_parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        return False, f"Найдено только {len(ids) if ids is not None else 0} маркеров. Нужно минимум 4!", None

    # Проверяем, что есть все 4 маркера (IDs: 0, 1, 2, 3)
    required_ids = [0, 1, 2, 3]
    found_ids = ids.flatten()

    missing_ids = [i for i in required_ids if i not in found_ids]
    if missing_ids:
        return False, f"Не найдены маркеры с ID: {missing_ids}", None

    # Сортируем маркеры по ID
    marker_corners = {}
    for i, marker_id in enumerate(found_ids):
        marker_corners[marker_id] = corners[i][0]

    # Определяем углы рабочей области (порядок: top-left, top-right, bottom-right, bottom-left)
    src_points = np.array([
        marker_corners[0][0],  # Top-left (ID 0)
        marker_corners[1][0],  # Top-right (ID 1)
        marker_corners[3][0],  # Bottom-right (ID 3)
        marker_corners[2][0],  # Bottom-left (ID 2)
    ], dtype=np.float32)

    # Вычисляем размеры рабочей области в пикселях
    width_top = np.sqrt(((src_points[1][0] - src_points[0][0]) ** 2) +
                        ((src_points[1][1] - src_points[0][1]) ** 2))
    width_bottom = np.sqrt(((src_points[3][0] - src_points[2][0]) ** 2) +
                           ((src_points[3][1] - src_points[2][1]) ** 2))
    height_left = np.sqrt(((src_points[2][0] - src_points[0][0]) ** 2) +
                          ((src_points[2][1] - src_points[0][1]) ** 2))
    height_right = np.sqrt(((src_points[3][0] - src_points[1][0]) ** 2) +
                           ((src_points[3][1] - src_points[1][1]) ** 2))

    max_width = max(int(width_top), int(width_bottom))
    max_height = max(int(height_left), int(height_right))

    # Целевые точки (прямоугольник)
    dst_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Вычисляем матрицу перспективы
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Вычисляем масштаб (пиксели/мм)
    marker_widths = []
    marker_heights = []
    for marker_id in required_ids:
        corner = marker_corners[marker_id]
        width = np.sqrt(((corner[1][0] - corner[0][0]) ** 2) +
                        ((corner[1][1] - corner[0][1]) ** 2))
        height = np.sqrt(((corner[2][0] - corner[1][0]) ** 2) +
                         ((corner[2][1] - corner[1][1]) ** 2))
        marker_widths.append(width)
        marker_heights.append(height)

    avg_marker_size_px = (np.mean(marker_widths) + np.mean(marker_heights)) / 2
    scale_factor = avg_marker_size_px / ARUCO_MARKER_SIZE_MM

    calibration_data = {
        'perspective_matrix': perspective_matrix,
        'scale_factor': scale_factor,
        'image_size': (max_width, max_height),
        'marker_corners': marker_corners,
        'num_markers_found': len(ids)
    }

    return True, f"Калибровка успешна! Найдено {len(ids)} маркеров. Масштаб: {scale_factor:.2f} пикс/мм", calibration_data


def apply_calibration(image, calibration_data):
    """
    Применяет коррекцию перспективы к изображению
    """
    perspective_matrix = calibration_data['perspective_matrix']
    img_size = calibration_data['image_size']

    # Коррекция перспективы
    calibrated_image = cv2.warpPerspective(
        image,
        perspective_matrix,
        img_size,
        flags=cv2.INTER_LINEAR
    )

    return calibrated_image


# =============================================================================
# ЛОГИКА ОБРАБОТКИ
# =============================================================================
def analyze_all_objects_logic(img_path, threshold_val, center_dist_threshold=50):
    """
    Возвращает: (processed_image_cv2, success_message, report_text)
    """
    # 1. Загрузка изображения
    image = cv2.imread(img_path)
    if image is None:
        return None, "Ошибка: Не удалось загрузить изображение.", ""

    original_image = image.copy()

    # 2. КАЛИБРОВКА С ARUCO МАРКЕРАМИ
    success, calib_message, calibration_data = calibrate_with_aruco(image)

    if not success:
        # Показываем изображение с найденными маркерами (даже если их < 4)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        detector_parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, detector_parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(image, corners, ids)

        return None, calib_message, f"ОШИБКА КАЛИБРОВКИ: {calib_message}"

    # Применяем коррекцию перспективы
    calibrated_image = apply_calibration(original_image, calibration_data)

    # Работаем с откалиброванным изображением
    img_gray = cv2.cvtColor(calibrated_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, threshold_val, 255, cv2.THRESH_BINARY)

    # 3. Обнаружение контуров
    contours, hierarchy = cv2.findContours(
        image=thresh,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    image_copy = calibrated_image.copy()
    all_ellipses = []

    # 4. Сбор всех подходящих эллипсов
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            all_ellipses.append(ellipse)

    if len(all_ellipses) < 2:
        return None, "Не удалось найти достаточно эллипсов. Попробуйте изменить порог.", ""

    # 5. Группировка эллипсов по объектам
    objects = []
    for ellipse in all_ellipses:
        (cx, cy), (w, h), angle = ellipse
        found_object = False

        # Пропускаем маркеры (если эллипс близко к маркеру)
        is_marker = False
        for marker_id, corners in calibration_data['marker_corners'].items():
            marker_center = np.mean(corners, axis=0)
            distance = np.sqrt((cx - marker_center[0]) ** 2 + (cy - marker_center[1]) ** 2)
            if distance < 50:
                is_marker = True
                break

        if is_marker:
            continue

        for obj in objects:
            ref_ellipse = obj[0]
            (ref_cx, ref_cy), _, _ = ref_ellipse
            distance = np.sqrt((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2)

            if distance < center_dist_threshold:
                obj.append(ellipse)
                found_object = True
                break

        if not found_object:
            objects.append([ellipse])

    # 6. Формирование отчета
    scale_factor = calibration_data['scale_factor']

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"ОТЧЕТ ОТ: {datetime.datetime.now().strftime('%H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append(f"Порог бинаризации: {threshold_val}")
    report_lines.append(f"ArUco маркеров найдено: {calibration_data['num_markers_found']}")
    report_lines.append(f"Масштаб: {scale_factor:.3f} пикс/мм")
    report_lines.append(f"Размер откалиброванного изображения: {calibration_data['image_size']}")
    report_lines.append(f"Всего найдено эллипсов: {len(all_ellipses)}")
    report_lines.append(f"Всего найдено объектов: {len(objects)}")
    report_lines.append("-" * 60)

    # Визуализация маркеров на итоговом изображении
    for marker_id, corners in calibration_data['marker_corners'].items():
        corners_int = corners.astype(np.int32)
        cv2.polylines(image_copy, [corners_int], True, (0, 255, 0), 2)
        cv2.putText(image_copy, f"M{marker_id}", (int(corners[0][0]), int(corners[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    colors = [
        (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128),
    ]
    color_names = ["Синий", "Красный", "Зеленый", "Голубой", "Пурпурный", "Желтый", "Фиолетовый", "Оранжевый"]

    for obj_idx, obj_ellipses in enumerate(objects):
        obj_ellipses.sort(key=lambda x: max(x[1]), reverse=True)

        if len(obj_ellipses) < 2:
            report_lines.append(f"[Объект #{obj_idx + 1}] - ПРОПУЩЕН (найден 1 контур)")
            continue

        outer_ellipse = obj_ellipses[0]
        inner_ellipse = obj_ellipses[1]

        (outer_w, outer_h) = outer_ellipse[1]
        (inner_w, inner_h) = inner_ellipse[1]

        outer_radius_px = max(outer_w, outer_h) / 2
        inner_radius_px = max(inner_w, inner_h) / 2
        wall_thickness_px = outer_radius_px - inner_radius_px

        # РЕАЛЬНЫЕ РАЗМЕРЫ В ММ
        outer_radius_mm = outer_radius_px / scale_factor
        inner_radius_mm = inner_radius_px / scale_factor
        wall_thickness_mm = wall_thickness_px / scale_factor

        (center_x, center_y), _, _ = outer_ellipse

        color_idx = obj_idx % len(colors)

        # Запись в отчет
        report_lines.append(f"\n[Объект #{obj_idx + 1}] ({color_names[color_idx]})")
        report_lines.append(f"  Центр: ({center_x:.1f}, {center_y:.1f})")
        report_lines.append(f"  Внешний димаетр: {outer_radius_px:.2f} пикс. / {outer_radius_mm:.2f} мм")
        report_lines.append(f"  Внутренний диаметр: {inner_radius_px:.2f} пикс. / {inner_radius_mm:.2f} мм")
        report_lines.append(f"  Толщина стенки: {wall_thickness_px:.2f} пикс. / {wall_thickness_mm:.2f} мм")

        # Визуализация
        color = colors[color_idx]
        cv2.ellipse(image_copy, outer_ellipse, color, 2, cv2.LINE_AA)
        cv2.ellipse(image_copy, inner_ellipse, color, 2, cv2.LINE_AA)

        # Номер объекта над кольцом
        label_x = int(center_x)
        label_y = int(center_y - outer_radius_px - 10)
        cv2.putText(image_copy, f"#{obj_idx + 1}", (label_x - 15, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Размеры рядом с объектом
        cv2.putText(image_copy, f"{outer_radius_mm:.1f}mm",
                    (int(center_x) - 30, int(center_y) + int(outer_radius_px) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    report_lines.append("\n" + "=" * 60)
    report_lines.append("Анализ завершен успешно!")
    report_lines.append("=" * 60)

    # Информация на изображении
    cv2.putText(image_copy, f"Objects: {len(objects)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image_copy, f"Scale: {scale_factor:.2f} px/mm", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    report_text = "\n".join(report_lines)
    return image_copy, f"Найдено объектов: {len(objects)}. Калибровка успешна!", report_text


# =============================================================================
# ИНТЕРФЕЙС ПРИЛОЖЕНИЯ
# =============================================================================
class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор объектов с ArUco калибровкой")
        self.root.geometry("1400x900")
        self.current_image_path = None
        self.processed_image_tk = None

        # --- Верхняя панель ---
        top_frame = tk.Frame(root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        settings_frame = tk.LabelFrame(top_frame, text="Параметры", padx=10, pady=5)
        settings_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(settings_frame, text="Порог (0-255): ").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.entry_threshold = tk.Entry(settings_frame, width=10)
        self.entry_threshold.insert(0, "170")
        self.entry_threshold.grid(row=0, column=1, padx=5)

        tk.Label(settings_frame, text="Дистанция центров: ").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.entry_center_dist = tk.Entry(settings_frame, width=10)
        self.entry_center_dist.insert(0, "50")
        self.entry_center_dist.grid(row=0, column=3, padx=5)

        tk.Label(settings_frame, text="Размер маркера (мм): ").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.entry_marker_size = tk.Entry(settings_frame, width=10)
        self.entry_marker_size.insert(0, "23.0")
        self.entry_marker_size.grid(row=0, column=5, padx=5)

        btn_frame = tk.Frame(top_frame, padx=10)
        btn_frame.pack(side=tk.RIGHT)

        self.btn_select = tk.Button(btn_frame, text="1. Выбрать изображение", command=self.select_image, width=20,
                                    bg="#ddd")
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.btn_process = tk.Button(btn_frame, text="2. Обработать", command=self.run_script, width=15,
                                     state=tk.DISABLED, bg="#add8e6")
        self.btn_process.pack(side=tk.LEFT, padx=5)

        # --- Центральная часть ---
        center_frame = tk.Frame(root)
        center_frame.pack(pady=10, expand=True, fill=tk.BOTH)

        # Слева: Изображение
        self.canvas_frame = tk.Frame(center_frame)
        self.canvas_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#f0f0f0")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar_y = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.scrollbar_x = tk.Scrollbar(center_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        self.image_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor=tk.NW)

        self.image_label = tk.Label(self.image_frame, text="Изображение не выбрано", bg="#f0f0f0")
        self.image_label.pack()
        self.image_frame.bind("<Configure>", self.on_frame_configure)

        # Справа: Результаты
        results_frame = tk.LabelFrame(center_frame, text="Результаты анализа", padx=10, pady=10)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)

        self.results_text = tk.Text(results_frame, width=50, height=35, font=("Consolas", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        results_scrollbar = tk.Scrollbar(results_frame, command=self.results_text.yview)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)

        self.btn_clear = tk.Button(results_frame, text="Очистить отчет", command=self.clear_results)
        self.btn_clear.pack(pady=5)

        # --- Статус бар ---
        self.status_label = tk.Label(root, text="Готов к работе", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self.current_image_path = file_path
            self.show_image(file_path)
            self.btn_process.config(state=tk.NORMAL)
            self.status_label.config(text=f"Выбран: {os.path.basename(file_path)}")
            self.clear_results()
            print(f"[{datetime.datetime.now()}] Выбран файл: {file_path}")

    def show_image(self, path, cv2_image=None):
        try:
            if cv2_image is not None:
                img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_rgb)
            else:
                img = Image.open(path)

            self.processed_image_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.processed_image_tk, text="")
            self.image_label.image = self.processed_image_tk
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except Exception as e:
            messagebox.showerror("Ошибка отображения", str(e))

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Ожидание обработки...\n")

    def run_script(self):
        if not self.current_image_path:
            return

        try:
            threshold_val = int(self.entry_threshold.get())
            center_dist = int(self.entry_center_dist.get())
            marker_size = float(self.entry_marker_size.get())

            # Обновляем глобальную константу
            global ARUCO_MARKER_SIZE_MM
            ARUCO_MARKER_SIZE_MM = marker_size

        except ValueError:
            messagebox.showerror("Ошибка параметров", "Параметры должны быть числами.")
            return

        print("-" * 30)
        print(f"[{datetime.datetime.now()}] ЗАПУСК СКРИПТА ОБРАБОТКИ...")

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Обработка...\n")
        self.root.update()

        try:
            processed_img_cv2, message, report_text = analyze_all_objects_logic(
                self.current_image_path,
                threshold_val,
                center_dist
            )

            if processed_img_cv2 is not None:
                self.show_image(None, cv2_image=processed_img_cv2)
                self.status_label.config(text="Обработка завершена!")
                messagebox.showinfo("Успех", message)

                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, report_text)

                print(report_text)
            else:
                self.status_label.config(text="Ошибка обработки")
                messagebox.showwarning("Внимание", message)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"ОШИБКА: {message}\n")
                print(f"ОШИБКА: {message}")

        except Exception as e:
            print(f"ОШИБКА ПРИ ОБРАБОТКЕ: {e}")
            messagebox.showerror("Критическая ошибка", str(e))
            self.results_text.insert(tk.END, f"\nКРИТИЧЕСКАЯ ОШИБКА: {str(e)}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()