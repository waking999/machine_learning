class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2:
            return False

        stack = []
        pair = {')': '(', '}': '{', ']': '['}
        for i in s:
            if i == '(' or i == '[' or i == '{':
                stack.append(i)

            elif i == ')' or i == ']' or i == '}':
                if len(stack) < 1:
                    return False

                if stack[-1] != pair[i]:
                    return False
                stack.pop()

        return len(stack) == 0
