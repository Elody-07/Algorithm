'''
皇后能横向，纵向和斜向移动，在这三条线上的其他棋子都可以被吃掉。所谓八皇后问题就是：将八位皇后放在一张8x8的棋盘上，使得每位皇后都无法吃掉别的皇后，（即任意两个皇后都不在同一条横线，竖线和斜线上），问一共有多少种摆法。
'''
# 解法一：回溯法，逐个确定皇后的位置
def queen(n):
    if n <= 0:
        return []
    res = []
    arr = [None] * n
    queen_core(n, arr, 0, res)
    return res

def queen_core(n, arr, col, res):
    if col == n:
        res.append(copy(arr))
        return
    for row in range(n):
        arr[col] = row
        flag = True
        for i in range(col):
            if arr[i] == row or abs(arr[i] - arr[col]) == col - i:
                flag = False
                break
        if flag:
            queen_core(n, arr, col+1, res)

print(queen(3))
print(queen(4))
print(len(queen(8)))

# 解法二：全排列，将满足条件的排列方式保存下来
def queen_2(n):
    if n <= 0:
        return []

    arr = list(range(n))
    res = []
    queen_2_core(arr, 0, res)
    return res

def queen_2_core(arr, begin, res):
    n = len(arr)
    if begin == n - 1:
        flag = True
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if abs(arr[i] - arr[j]) == abs(i-j):
                    flag = False
                    break
        if flag:
            res.append(copy(arr))
        return

    for i in range(begin, n):
        temp = arr[i]
        arr[i] = arr[begin]
        arr[begin] = temp
        queen_2_core(arr, begin+1, res)
        temp = arr[i]
        arr[i] = arr[begin]
        arr[begin] = temp

print(queen_2(3))
print(queen_2(4))
print(len(queen_2(8)))


'''
给你六种面额 1、5、10、20、50、100 元的纸币，假设每种币值的数量都足够多，编写程序求组成N元（N为0~10000的非负整数）的不同组合的个数。
时间复杂度O(kn)
'''

def ComposeN(a, n):
    dp = [0] * (n+1)
    dp[0] = 1
    for item in a:
        for j in range(item, n+1):
            dp[j] = dp[j] + dp[j - item]
    return dp[n]

a = [1, 5, 10, 20, 50, 100]
for i in range(10, 20):
    print(i, ComposeN(a, i))