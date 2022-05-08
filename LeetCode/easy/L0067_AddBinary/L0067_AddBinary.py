class Solution:
    def addBinary(self, a: str, b: str) -> str:
        a = a[::-1]
        b = b[::-1]
        carrier = 0
        rtn = []
        len_a = len(a)
        len_b = len(b)
        len_ab = min(len_a, len_b)
        for i in range(len_ab):
            t1 = int(a[i])
            t2 = int(b[i])
            t = t1 + t2 + carrier
            if t > 1:
                carrier = 1
                rtn.append(str(t%2))
            else:
                carrier = 0
                rtn.append(str(t))

        if len_a > len_ab:
            for i in range(len_ab, len_a):
                t1 = int(a[i])
                t = t1 + carrier
                if t >1 :
                    carrier = 1
                    rtn.append(str(t%2))
                else:
                    carrier = 0
                    rtn.append(str(t))
        else:
            for i in range(len_ab, len_b):
                t1 = int(b[i])
                t = t1 + carrier
                if t > 1:
                    carrier = 1
                    rtn.append(str(t%2))
                else:
                    carrier = 0
                    rtn.append(str(t))

        if carrier == 1:
            rtn.append("1")
        rtn.reverse()

        return ''.join(rtn)
