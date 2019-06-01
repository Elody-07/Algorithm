'''
皇后能横向，纵向和斜向移动，在这三条线上的其他棋子都可以被吃掉。所谓八皇后问题就是：将八位皇后放在一张8x8的棋盘上，使得每位皇后都无法吃掉别的皇后，（即任意两个皇后都不在同一条横线，竖线和斜线上），问一共有多少种摆法。
'''
def queen(arr):
    res = {
        'count': 0
    }
    queen_core(arr, res, cur = 0)
    return res['count'] 

def queen_core(arr, res, cur=0):
    # arr是一个长度为8的list，代表棋盘的八列，其中每个值代表棋盘的行数
    if cur == len(arr):
        res['count'] += 1
        print(arr)
        return  
    
    for row in range(len(arr)):
        arr[cur], flag = row, True 
        # check
        for col in range(cur): 
            # 两个皇后在一条斜线上判断依据：行差=列差
            if arr[col] == row or abs(arr[col] - row) == cur - col:
                flag = False
                break
        if flag:
            queen_core(arr, res , cur+1)

a = queen([None] * 4)
print(a)