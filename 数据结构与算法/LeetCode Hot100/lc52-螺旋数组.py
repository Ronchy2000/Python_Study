matrix = [[1,2,3],
          [4,5,6],
          [7,8,9],
          [10,11,12]]
print(matrix)
print(matrix[0][:])
print(len(matrix))

class Solution:
    def func(self, matrix):
        if not matrix:
            return []
        l, r, t, b = 0, len(matrix[0]) - 1, 0, len(matrix) - 1
        result = []
        while True:
            for i in range(l, r+1): # 左 -> 右
                result.append(matrix[t][i])
            t += 1
            if l > r: break

            for i in range(t, b+1): # 上 -> 下
                result.append(matrix[i][r])
            r -= 1
            if t > b: break

            for i in range(r, l-1, -1): # 右 -> 左
                result.append(matrix[b][i])
            b -= 1
            if l > r: break

            for i in range(b, t-1, -1): # 下 -> 上
                result.append(matrix[i][l])
            l += 1
            if t > b: break
        return result
if __name__ == "__main__":
    sol = Solution()
    print(sol.func(matrix))