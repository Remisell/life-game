import taichi as ti

config = {
    'screen_width': 1920,
    'screen_height': 1080,
    'cell_size': 1,
}


def main():
    SCREEN_WIDTH = config['screen_width']
    SCREEN_HEIGHT = config['screen_height']
    CELL_SIZE = config['cell_size']

    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

    try:
        ti.init(arch=ti.gpu)
    except:
        ti.init(arch=ti.cpu)

    # Поля для клеток
    cells = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT))
    cells_old = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT))

    # Поле для рендеринга
    image_field = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_WIDTH, GRID_HEIGHT))

    # Для живой клетки: правила выживания (сколько соседей нужно, чтобы остаться живой)
    survival_rules = ti.field(dtype=ti.i32, shape=9)
    # Для мертвой клетки: правила рождения (сколько соседей нужно, чтобы родиться)
    birth_rules = ti.field(dtype=ti.i32, shape=9)

    # Инициализация правил по умолчанию
    @ti.kernel
    def init_default_rules():
        # Классические правила: выживание при 2 или 3 соседях
        for i in range(9):
            survival_rules[i] = 0
            birth_rules[i] = 0
        survival_rules[2] = 1
        survival_rules[3] = 1
        birth_rules[3] = 1

    init_default_rules()

    @ti.kernel
    def init():
        for i, j in cells:
            cells[i, j] = 0

    @ti.kernel
    def randomize(probability: ti.f32):
        for i, j in cells:
            if ti.random() < probability:
                cells[i, j] = 1
            else:
                cells[i, j] = 0

    @ti.kernel
    def clear():
        for i, j in cells:
            cells[i, j] = 0

    @ti.kernel
    def update():
        # Копируем текущее состояние
        for i, j in cells:
            cells_old[i, j] = cells[i, j]

        # Обновляем клетки по правилам
        for i, j in cells:
            neighbors = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx = i + dx
                    ny = j + dy
                    if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                        neighbors += cells_old[nx, ny]

            if cells_old[i, j] == 1:
                # Проверяем правила выживания
                if survival_rules[neighbors] == 1:
                    cells[i, j] = 1
                else:
                    cells[i, j] = 0
            else:
                # Проверяем правила рождения
                if birth_rules[neighbors] == 1:
                    cells[i, j] = 1
                else:
                    cells[i, j] = 0

    @ti.kernel
    def set_cell(x: ti.i32, y: ti.i32, state: ti.i32):
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            cells[x, y] = state

    @ti.kernel
    def toggle_cell(x: ti.i32, y: ti.i32):
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            cells[x, y] = 1 - cells[x, y]

    @ti.kernel
    def render():
        for i, j in image_field:
            if cells[i, j] == 1:
                # Живая клетка
                image_field[i, j] = [1.0, 1.0, 1.0]
            else:
                # Мертвая клетка
                image_field[i, j] = [0.0, 0.0, 0.0]

    @ti.kernel
    def count_population() -> ti.i64:
        count = 0
        for i, j in cells:
            count += cells[i, j]
        return count

    @ti.kernel
    def set_rule(rule_type: ti.i32, neighbor_count: ti.i32, value: ti.i32):
        if rule_type == 0:
            survival_rules[neighbor_count] = value
        else:
            birth_rules[neighbor_count] = value

    @ti.kernel
    def get_survival_rule(neighbor_count: ti.i32) -> ti.i32:
        return survival_rules[neighbor_count]

    @ti.kernel
    def get_birth_rule(neighbor_count: ti.i32) -> ti.i32:
        return birth_rules[neighbor_count]

    def settings(gui, speed, survival, birth, paused):
        gui.begin("Settings", 0.05, 0.05, 0.3, 0.85)

        gui.text("=== Simulation ===")
        gui.text("Update Speed:")
        new_speed = gui.slider_int("##speed", speed, 1, 30)

        gui.text("")
        gui.text("=== Game Rules ===")
        gui.text("Survival rules (live cell):")

        # Создаем чекбоксы для правил выживания
        for i in range(0, 9):
            val = gui.checkbox(f"S {i}", survival[i])
            if val != survival[i]:
                survival[i] = val
                set_rule(0, i, 1 if val else 0)

        gui.text("Birth rules (dead cell):")
        # Создаем чекбоксы для правил рождения
        for i in range(0, 9):
            val = gui.checkbox(f"B {i}", birth[i])
            if val != birth[i]:
                birth[i] = val
                set_rule(1, i, 1 if val else 0)

        gui.text("")
        gui.text("=== Presets ===")
        if gui.button("Conway's Life (B3/S23)"):
            # B3/S23
            for i in range(9):
                survival[i] = (i == 2 or i == 3)
                birth[i] = (i == 3)
                set_rule(0, i, 1 if survival[i] else 0)
                set_rule(1, i, 1 if birth[i] else 0)

        if gui.button("HighLife (B36/S23)"):
            # B36/S23
            for i in range(9):
                survival[i] = (i == 2 or i == 3)
                birth[i] = (i == 3 or i == 6)
                set_rule(0, i, 1 if survival[i] else 0)
                set_rule(1, i, 1 if birth[i] else 0)

        if gui.button("Seeds (B2/S)"):
            # B2/S
            for i in range(9):
                survival[i] = False
                birth[i] = (i == 2)
                set_rule(0, i, 0)
                set_rule(1, i, 1 if birth[i] else 0)

        if gui.button("Life Without Death (B3/S012345678)"):
            # B3/S012345678
            for i in range(9):
                survival[i] = True
                birth[i] = (i == 3)
                set_rule(0, i, 1)
                set_rule(1, i, 1 if birth[i] else 0)

        if gui.button("2x2 (B36/S125)"):
            # B36/S125
            for i in range(9):
                survival[i] = (i == 1 or i == 2 or i == 5)
                birth[i] = (i == 3 or i == 6)
                set_rule(0, i, 1 if survival[i] else 0)
                set_rule(1, i, 1 if birth[i] else 0)

        gui.text("")
        gui.text("=== Controls ===")
        if gui.button("Randomize"):
            randomize(0.3)
        if gui.button("Clear All"):
            clear()

        pause_text = "Resume" if paused else "Pause"
        if gui.button(pause_text):
            paused = not paused

        gui.end()
        return new_speed, survival, birth, paused


    init()
    randomize(0.1)

    # Создание окна
    window = ti.ui.Window("Game of Life", (SCREEN_WIDTH, SCREEN_HEIGHT),
                          vsync=True, show_window=True)

    canvas = window.get_canvas()
    gui = window.get_gui()
    canvas.set_background_color((0, 0, 0))

    # Параметры
    simulation_speed = 10
    show_settings = False
    generation = 0
    paused = False

    # Состояния правил для отображения в GUI
    survival_states = [False] * 9
    birth_states = [False] * 9

    # Инициализация состояний правил с помощью отдельных функций
    for i in range(9):
        survival_states[i] = (get_survival_rule(i) == 1)
        birth_states[i] = (get_birth_rule(i) == 1)

    # Состояния клавиш
    previous_c_state = False
    previous_r_state = False
    previous_space_state = False
    previous_s_state = False

    # Основной цикл
    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):
            break

        mouse_pos = window.get_cursor_pos()
        grid_x = int(mouse_pos[0] * GRID_WIDTH)
        grid_y = int(mouse_pos[1] * GRID_HEIGHT)
        grid_x = max(0, min(grid_x, GRID_WIDTH - 1))
        grid_y = max(0, min(grid_y, GRID_HEIGHT - 1))

        # Обработка клавиш
        current_c_state = window.is_pressed('c') or window.is_pressed('C')
        if current_c_state and not previous_c_state:
            show_settings = not show_settings
        previous_c_state = current_c_state

        current_r_state = window.is_pressed('r') or window.is_pressed('R')
        if current_r_state and not previous_r_state:
            randomize(0.3)
        previous_r_state = current_r_state

        current_space_state = window.is_pressed(ti.ui.SPACE)
        if current_space_state and not previous_space_state:
            paused = not paused
        previous_space_state = current_space_state

        current_s_state = window.is_pressed('s') or window.is_pressed('S')
        if current_s_state and not previous_s_state:
            # Ручное обновление на один шаг
            if paused:
                update()
                generation += 1
        previous_s_state = current_s_state

        if show_settings:
            # Отображаем настройки
            new_speed, survival_states, birth_states, paused = settings(
                gui, simulation_speed, survival_states, birth_states, paused
            )

            # Обновляем значения
            simulation_speed = new_speed
        else:
            # Отображаем подсказки
            gui.begin("Controls", 0.7, 0.05, 0.25, 0.3)
            gui.text(f"Generation: {generation}")
            gui.text(f"Population: {count_population()}")

            # Отображаем текущие правила
            survival_str = ''.join(str(i) for i in range(9) if survival_states[i])
            birth_str = ''.join(str(i) for i in range(9) if birth_states[i])
            gui.text(f"Rules: B{birth_str if birth_str else '0'}/S{survival_str if survival_str else '0'}")

            gui.text("")
            gui.text("LMB: Toggle cell")
            gui.text("RMB: Clear area")
            gui.text("SPACE: Pause/Resume")
            gui.text("S: Step (when paused)")
            gui.text("C: Settings")
            gui.text("R: Randomize")
            gui.text("ESC: Quit")
            gui.end()

            # Рисование клеток мышью
            if window.is_pressed(ti.ui.LMB):
                set_cell(grid_x, grid_y, 1)

            if window.is_pressed(ti.ui.RMB):
                set_cell(grid_x, grid_y, 0)

        # Обновление симуляции
        if not paused:
            # Фиксированное количество обновлений в секунду
            frame_updates = max(1, simulation_speed // 10)
            for _ in range(frame_updates):
                update()
                generation += 1

        render()
        canvas.set_image(image_field)
        window.show()


if __name__ == '__main__':
    main()
