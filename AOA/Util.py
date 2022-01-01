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
