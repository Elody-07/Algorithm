arr1 = []
arr2 = [4,2,5,1,8,2,8]
arr3 = [1,5,3,2,6,4,7]
arr4 = [1,2,3,4,5,6,7]
arr5 = [7,6,5,4,3,2,1]

def InsertSort(arr):
    '''
    直接插入排序
    '''
    if len(arr) == 0:
        return False
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            temp = arr[i]
            j = i-1
            while temp < arr[j] and j >= 0:
            # 由于list索引可为负，故要加上j>=0的条件
                arr[j+1] = arr[j]
                j -= 1
            arr[j + 1] = temp
    return arr

# print(InsertSort(arr1))
# print(InsertSort(arr2))
# print(InsertSort(arr3))
# print(InsertSort(arr4))
# print(InsertSort(arr5))


def BubbleSort(arr):
    '''
    冒泡排序
    '''
    if len(arr) == 0:
        return False 
    for i in range(1, len(arr)): # 第i趟
        for j in range(0, len(arr)-i ): # 无序区
            if arr[j] > arr[j+1]:
                temp = arr[j]
                arr[j] = arr[j+1]
                arr[j+1] = temp
    return arr

# print(BubbleSort(arr1))
# print(BubbleSort(arr2))
# print(BubbleSort(arr3))
# print(BubbleSort(arr4))
# print(BubbleSort(arr5))

def BetterBubbleSort(arr):
    '''
    改进冒泡排序(缩小无序元素范围（尾部）)
    '''
    if len(arr) == 0:
        return False
    pos = len(arr)-1 # 初始无序元素的范围
    while(pos != 0): # 外循环，减少趟数
        bound = pos 
        pos = 0
        for i in range(0, bound):
            if arr[i] > arr[i+1]:
                temp = arr[i]
                arr[i] = arr[i+1]
                arr[i+1] = temp
                pos = i

    return arr

# print(BetterBubbleSort(arr1))
# print(BetterBubbleSort(arr2))
# print(BetterBubbleSort(arr3))
# print(BetterBubbleSort(arr4))
# print(BetterBubbleSort(arr5))
                            

import random
def Partition(arr):
    start = 0
    end = len(arr) - 1
    center_idx = random.randint(start, end)

    center = arr[center_idx] # 将轴值放在数组第一个
    arr[center_idx] = arr[start]
    arr[start] = center 

    while(start < end):
        while arr[end] >= center:
            end -= 1
        arr[start] = arr[end]
        while arr[start] <= center:
            start += 1
        arr[end] = arr[start]
    assert start==end 
    arr[start] = arr[0]
    return start

def QuickSort(arr):
    '''
    快速排序
    '''
    if len(arr) == 0:
        return False
    if len(arr) == 1:
        return arr
    middle = Partition(arr)
    QuickSort(arr[0 : middle])
    QuickSort(arr[middle + 1 :])

# print(BetterBubbleSort(arr1))
# print(BetterBubbleSort(arr2))
# print(BetterBubbleSort(arr3))
# print(BetterBubbleSort(arr4))
# print(BetterBubbleSort(arr5))


def SelectSort(arr):
    '''
    简单选择排序
    '''
    if len(arr) == 0:
        return False 
    for i in range(0, len(arr)-1):
        index = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[index]:
                index = j
        if index != i:
            temp = arr[index]
            arr[index] = arr[i]
            arr[i] = temp
    return arr

# print(SelectSort(arr1))
# print(SelectSort(arr2))
# print(SelectSort(arr3))
# print(SelectSort(arr4))
# print(SelectSort(arr5))


def Sift(arr, i, n):
    j = 2 * i
    while(j <= n):
        if j < n and arr[j] < arr[j+1]:
            j += 1
        if arr[i] > arr[j]:
            break
        else:
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
            i = j 
            j = 2 * i
    return arr 

def HeapSort(arr):
    '''
    堆排序
    '''
    n = len(arr) - 1
    for i in range(n//2, 0, -1):
        Sift(arr, i, n)
    for i in range(n, 1, -1):
        temp = arr[1]
        arr[1] = arr[i]
        arr[i] = temp
        Sift(arr, 1, i-1)
    return arr


h1 = [-1]
h2 = [-1, 4,2,5,1,8,2,8]
h3 = [-1, 1,5,3,2,6,4,7]
h4 = [-1, 1,2,3,4,5,6,7]
h5 = [-1, 7,6,5,4,3,2,1]
# print(HeapSort(h1))
# print(HeapSort(h2))
# print(HeapSort(h3))
# print(HeapSort(h4))
# print(HeapSort(h5))



# 归并两个相邻序列 arr1[s]~arr1[m], arr1[m+1]~arr1[t] 
def Merge(arr1, s, m, t):
    arr2 = []
    i = s # i 指向 r[s]~r[m]
    j = m+1 # j 指向 r[m+1]~r[t]
    while (i <= m and j <= t):
        if arr1[i] <= arr1[j]:
            arr2.append(arr1[i])
            i += 1
        else:
            arr2.append(arr1[j])
            j += 1
    while i <= m:
        arr2.append(arr1[i])
        i += 1
    while j <= t:
        arr2.append(arr1[j])
        j += 1
    return arr2

# 一趟排序
def MergePass(arr1, n, h): # n是元素个数，最后一个元素下标n-1, h是每次归并子序列的个数
    arr2 = []
    i = 0
    while(i < n-2*h): # 剩下元素 大于2h个
        arr2 += Merge(arr1, i, i+h-1, i+2*h-1)
        i += 2*h
    if i < n-h: # 剩下的元素 大于h小于2h个
        arr2 += Merge(arr1, i, i+h-1, n-1)
    if i >= n-h: # 剩下的元素 小于等于h个
        while i <= n-1:
            arr2.append(arr1[i])
            i += 1
    return arr2

def MergeSort(arr1):
    '''
    二路归并排序
    '''
    n = len(arr1)
    h = 1
    while(h < n):
        arr1 = MergePass(arr1, n, h)
        h = 2 * h
    return arr1

# print(MergeSort(arr1))
# print(MergeSort(arr2))
# print(MergeSort(arr3))
# print(MergeSort(arr4))
# print(MergeSort(arr5))

        


    