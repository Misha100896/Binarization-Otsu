# Binarization-Otsu

Для запуска нужно:
1) Чтобы на компьютере была собрана OpenCV
2) Настроить для проекта в Visual Studio пути к include и lib'ам
3) Указать в параметрах входной строки сначала какой файл хотим биноризовать, второй параметр куда хотим сохранить
4) Чтобы протестировать на предоставленных тестах, нужно указать директиву FOR_EXAMPLES, и положить картинки рядом с main.cpp

Итоги работы бинаризации: http://gg.gg/4jy61
Среднее время работы: 8.80407e-006(мсек/пиксель)

Итог:
Бинаризация Отсу хорошо работает, когда гистограмма яркости на изображении сильно варьируется. 
Однако когда на изображении мало текста, то даже маленькую тень, алгоритм переводит в черный цвет.