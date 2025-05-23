# Решатель задачи релаксации

Этот репозиторий содержит реализацию на Python численного решателя (солвера) для задачи релаксации уравнения Больцмана.
Решатель фокусируется на вычислении интеграла столкновений с использованием проекционного метода и сеток Коробова, 
как описано в Пособии.
Проект моделирует релаксацию неравновесного распределения газа к максвелловскому равновесному состоянию, обеспечивая 
сохранение массы, импульса и энергии, то есть консервативность системы
------------------
## В проекте используются 
- вместо привычного метода Монте-Карло **сетки Коробова** для вычисления интеграла 
столкновений
- консервативный **проекционный метод** (подход **Черемисина**) для аппроксимации интеграла столкновений, обеспечивая нулевой 
  вклад при максвелловском распределении.
- **кинетическое уравнение Больцмана** для пространственно однородного газа, моделирующее релаксацию к равновесию.
- **законы сохранения**:
  - макроскопических параметров (плотности, импульса, энергии).
  - симметрии интеграла столкновений.
  - сходимости по разрешению скоростной сетки.
  - скорости релаксации в сравнении с аналитической моделью.
- **модель твёрдых сфер**, то есть предполагается, что взаимодействия молекул следуют непроницаемому потенциалу (для 
  упрощения).
------------------
## Требования
- **Python 3.13+**
- **Зависимости**:
  - NumPy: Для численных вычислений и работы с массивами.
  - Matplotlib: Для построения графиков релаксации и сходимости.
  - Установка зависимостей:
    ```bash
    pip install numpy matplotlib
------------------
## Структура файлов и папок:

- `code/`
    - `data/` посчитанные данные
    - `sem_1.py`преобразование скоростей и переход к новым переменным
    - `sem_2.py`реализация проекционного метода
    - `sem_3.py`шаги релаксации по временной и скоростной сеткам
  - `example.ipynb` блокнот для моделирования релаксации и выполнения проверок
- `.gitignore` файл указывает намеренно не отслеживаемые файлы, которые git должен игнорировать
- `Пособие.pdf` теоретическое обоснование и алгоритм метода
- `README.md`: этот файл

------------------

## Краткое руководство
1. **Клонирование репозитория**:
   ```bash
   git clone https://github.com/physicistmaksim/compmathproject.git
   ```
2. **Описание работы**
   - Инициализируется скоростная сетка (20 узлов по каждому измерению с обрезанием $\xi_{\text{cut}} = 4.8$)
   - Генерируются сетки Коробова с использованием алгоритма из Семинара №3 с простыми числами (например, $p 
     = 4~000~037$) для эффективного интегрирования.
   - Задаются начальные функции распределения (например, $f_1$ или $f_2$ из уравнений 3.35, 3.36).
   Они нормируются на единичную плотность
   - Вычисляется интеграл столкновений на нескольких временных шагах ($\tau = 0.1$).

3. **Параметры**:
   - Размеры сетки ($N_{\xi_x}, N_{\xi_y}, N_{\xi_z}$). Для удобства они равны по каждой координате
   - Временной шаг ($\tau$) и общая длительности моделирования ($10~\tau$).
   - Начальное условие ($\tilde{T}$, например, 0.95 для слабого неравновесия).
   - Размера сетки Коробова ($p$, на основе $W_{\min}$ из уравнения 3.29).
------------------
## Команда
Над проектом работали студенты Б02-206 группы 

- **Максим Володин** - семинар 3 в пособии и методология
- **Иван Касьянов** - семинар 2 в пособии и отладка кода
- **Максим Мурашко** - семинар 1 в пособии и appearance

------------------
Проект распространяется под лицензией MIT.
По вопросам или предложениям создайте issue
