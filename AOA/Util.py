class Util:
    def __init__(self):
        return

    @staticmethod
    def get_decimal_length(number):
        # get decimal number length
        num_str = str(number)
        num_str_dot_place = num_str.index('.')
        return len(num_str[num_str_dot_place + 1:])

    @staticmethod
    def convert_1d_array_to_2d(array):
        rtn_array = []
        for item in array:
            rtn_array.append([item])

        return rtn_array

    @staticmethod
    def generate_curve_array_2d(array, step):
        rtn_array = []
        array_min = min(array)
        array_max = max(array)
        array_diff = array_max - array_min
        array_diff_inc = array_diff / step
        rtn_array.append(array_min)

        for i in range(1, step):
            rtn_array.append(rtn_array[i - 1] + array_diff_inc)

        return rtn_array
