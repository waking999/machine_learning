import math
import time


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.available = []
        self.value = 0

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")," + str(self.value) + "," + str(self.available)


# sudoku = [
#     [4, 0, 0, 0],
#     [0, 0, 0, 1],
#     [0, 0, 1, 0],
#     [0, 0, 3, 0]
# ]


sudoku = [
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0]
]

SUDOKU_SIZE = 9
SUDOKU_BLANK = 0


def init_point():
    global sudoku
    global SUDOKU_SIZE
    global SUDOKU_BLANK

    _point_index_dict = {}

    for i in range(SUDOKU_SIZE):
        for j in range(SUDOKU_SIZE):
            if sudoku[i][j] == SUDOKU_BLANK:
                _p = Point(i, j)
                for v in range(1, SUDOKU_SIZE + 1):
                    if v not in row_num(_p) and v not in col_num(_p) and v not in block_num(_p):
                        _p.available.append(v)

                ind = i * SUDOKU_SIZE + j
                _point_index_dict[ind] = _p

                if len(_p.available) == 1:
                    remove_point_available(_p, _point_index_dict, ind)

    return _point_index_dict


def remove_point_available(_p, _point_index_dict, ind):
    global sudoku

    _p.value = _p.available[0]
    sudoku[_p.x][_p.y] = _p.value
    _point_index_dict.pop(ind)
    remove_available(_p, _point_index_dict)


def remove_available(_p, _point_index_dict):
    global sudoku

    remove_available_row(_p, _point_index_dict)
    remove_available_col(_p, _point_index_dict)
    remove_available_block(_p, _point_index_dict)
    show_sudoku()


def remove_available_row(_p, _point_index_dict):
    global sudoku

    global SUDOKU_SIZE

    for j in range(SUDOKU_SIZE):
        if j != _p.y:
            dict_ind = _p.x * SUDOKU_SIZE + j
            if dict_ind in _point_index_dict:
                tmp_p = _point_index_dict[dict_ind]
                try:
                    tmp_p.available.remove(_p.value)
                except ValueError:
                    pass

                if len(tmp_p.available) == 1:
                    remove_point_available(tmp_p, _point_index_dict, dict_ind)
            else:
                continue
        else:
            continue


def remove_available_col(_p, _point_index_dict):
    global sudoku
    global SUDOKU_SIZE

    for i in range(SUDOKU_SIZE):
        if i != _p.x:
            ind = i * SUDOKU_SIZE + _p.y
            if ind in _point_index_dict:
                tmp_p = _point_index_dict[ind]
                try:
                    tmp_p.available.remove(_p.value)
                except ValueError:
                    pass
                if len(tmp_p.available) == 1:
                    remove_point_available(tmp_p, _point_index_dict, ind)
            else:
                continue
        else:
            continue


def remove_available_block(_p, _point_index_dict):
    global sudoku

    global SUDOKU_SIZE

    blockXStart, blockXEnd, blockYStart, blockYEnd = get_block(_p)
    for i in range(blockXStart, blockXEnd):
        if i != _p.x:
            for j in range(blockYStart, blockYEnd):
                if j != _p.y:
                    ind = i * SUDOKU_SIZE + j
                    if ind in _point_index_dict:
                        tmp_p = _point_index_dict[ind]
                        try:
                            tmp_p.available.remove(_p.value)
                        except ValueError:
                            pass
                        if len(tmp_p.available) == 1:
                            remove_point_available(tmp_p, _point_index_dict, ind)
                    else:
                        continue
                else:
                    continue
        else:
            continue


def check(_p):
    global sudoku
    global SUDOKU_BLANK
    global SUDOKU_SIZE

    if _p.value == SUDOKU_BLANK:
        return False

    if _p.value not in row_num(_p) and _p.value not in col_num(_p) and _p.value not in block_num(_p):
        return True

    return False


def try_insert(_p, _point_index_dict):
    global sudoku

    for v in _p.available:
        _p.value = v
        if check(_p):

            sudoku[_p.x][_p.y] = v
            if not bool(_point_index_dict):
                t2 = time.time()
                print("use time:" + str((t2 - t1) / 1000) + " s.")
                show_sudoku()
                exit()

            _p2_item = _point_index_dict.popitem()
            _p2_ind = _p2_item[0]
            _p2 = _p2_item[1]
            try_insert(_p2, _point_index_dict)
            _p2.value = SUDOKU_BLANK
            sudoku[_p2.x][_p2.y] = SUDOKU_BLANK
            _point_index_dict[_p2_ind] = _p2

            _p.value = SUDOKU_BLANK
            sudoku[_p.x][_p.y] = SUDOKU_BLANK


def row_num(_p):
    global sudoku

    row = set(sudoku[_p.x])
    row.remove(SUDOKU_BLANK)
    return row


def col_num(_p):
    global sudoku

    col = []
    for i in range(SUDOKU_SIZE):
        col.append(sudoku[i][_p.y])
    col = set(col)
    col.remove(SUDOKU_BLANK)
    return col


def block_num(_p):
    global sudoku
    global SUDOKU_SIZE
    global SUDOKU_BLANK

    blockXStart, blockXEnd, blockYStart, blockYEnd = get_block(_p)

    block = []
    for i in range(blockXStart, blockXEnd):
        for j in range(blockYStart, blockYEnd):
            block.append(sudoku[i][j])
    block = set(block)
    block.remove(SUDOKU_BLANK)
    return block


def get_block(_p):
    global SUDOKU_SIZE
    block_size = int(math.sqrt(SUDOKU_SIZE))
    blockXStart = (_p.x // block_size) * block_size
    blockXEnd = blockXStart + block_size
    blockYStart = (_p.y // block_size) * block_size
    blockYEnd = blockYStart + block_size

    return blockXStart, blockXEnd, blockYStart, blockYEnd


def show_sudoku():
    global sudoku
    global SUDOKU_SIZE
    for i in range(SUDOKU_SIZE):
        for j in range(SUDOKU_SIZE):
            print(sudoku[i][j], end=' ')
        print('')

    print('\n')


if __name__ == '__main__':
    t1 = time.time()

    show_sudoku()
    point_index_dict = init_point()
    if not bool(point_index_dict):
        t2 = time.time()
        print("use time:" + str((t2 - t1) / 1000) + " s.")
        show_sudoku()
        exit()

    p = point_index_dict.popitem()[1]
    try_insert(p, point_index_dict)


