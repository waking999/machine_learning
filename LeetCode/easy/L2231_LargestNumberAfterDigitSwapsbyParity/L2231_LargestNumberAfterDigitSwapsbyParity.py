class Solution:
    def largestInteger(self, num: int) -> int:
        num_list = list(map(lambda i: int(i), str(num)))
        odd_list = list(filter(lambda i: i % 2 == 1, num_list))
        even_list = list(filter(lambda i: i % 2 == 0, num_list))

        odd_list.sort()
        even_list.sort()

        rtn = ''
        for i in num_list:
            if i % 2 == 0:
                rtn += str(even_list.pop())
            else:
                rtn += str(odd_list.pop())

        return int(rtn)
